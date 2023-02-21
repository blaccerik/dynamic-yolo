import io
import zipfile
import tarfile
from io import BytesIO

import PIL
import torch
from flask import Blueprint, request, jsonify, send_file
from marshmallow import ValidationError
from werkzeug.utils import secure_filename

from project.schemas.zip_upload import ZipUpload
from project.services.file_upload_service import upload_files
from project.models.annotation import Annotation
from project.models.image import Image
from project.services.project_service import create_project, get_models, get_all_projects, get_project_info, \
    change_settings, get_model
from project.schemas.project import Project
from project.schemas.upload import Upload
from project.schemas.model import Model
from project.schemas.projectsettings import ProjectSettingsSchema

REQUEST_API = Blueprint('project', __name__, url_prefix="/projects")

image_types = [
    "image/jpeg",
    "image/png"
]

text_types = [
    "text/plain"
]

zip_types = [
    "application/gzip"
]


def _text_to_annotations(text, name):
    _list = []
    for line in text.splitlines():
        try:
            nr, x, y, w, h = line.strip().split(" ")
            nr = int(nr)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            # check ranges
            if nr < 0:
                return jsonify({'error': f"File: {name}.txt did not yolo format"}), 400
            if x > 1 or x < 0:
                return jsonify({'error': f"File: {name}.txt did not yolo format"}), 400
            if y > 1 or y < 0:
                return jsonify({'error': f"File: {name}.txt did not yolo format"}), 400
            if w > 1 or w < 0:
                return jsonify({'error': f"File: {name}.txt did not yolo format"}), 400
            if h > 1 or h < 0:
                return jsonify({'error': f"File: {name}.txt did not yolo format"}), 400
            _list.append((Annotation(x_center=x, y_center=y, width=w, height=h, class_id=nr), name))
        except Exception as e:
            return jsonify({'error': f"Can't read file: {name}.txt"}), 400
    return _list


def _check_files(files):
    final_files = []
    print(files)
    for file in files:
        content = file.stream.read()
        file.stream.close()
        filename = secure_filename(file.filename)
        print(filename)
        # db_objects = _bytes_to_db_object(content, secure_filename(file.filename))
        # if type(db_objects) is tuple:
        #     return db_objects
        # final_files.extend(db_objects)
    return final_files


def _check_zip_file(file):
    content = file.stream.read()
    file.stream.close()
    try:
        tar = tarfile.open(fileobj=BytesIO(content))
    except Exception:
        return jsonify({'error': f'tar file could not ne opened'}), 400
    files = []
    for item in tar:
        if not item.isfile():
            continue
        _bytes = tar.extractfile(item.name).read()
        db_objects = _bytes_to_db_object(_bytes, item.name)
        if type(db_objects) is tuple:
            return db_objects
        files.extend(db_objects)
    return files


def _bytes_to_db_object(_bytes, name: str):
    if name.endswith(".txt"):
        text = str(_bytes, "utf-8")
        name = name.split(".")[0]
        _list = _text_to_annotations(text, name)
        return _list
    elif name.endswith(".png") or name.endswith(".jpg"):  # image file
        io_bytes = BytesIO(_bytes)
        try:
            img = PIL.Image.open(io_bytes)
        except Exception:
            return jsonify({'error': f"Can't read file: {name}"}), 400
        name = name.split(".")[0]
        return [(Image(image=_bytes, width=img.size[0], height=img.size[1]), name)]
    else:
        return jsonify({'error': f"Can't read file: {name}"}), 400


@REQUEST_API.route('/', methods=['POST'])
def create_record():
    data = Project().load(request.json)
    name = data["name"]
    max_class_nr = data["max_class_nr"]
    code = create_project(name, max_class_nr)
    if code == -1:
        return jsonify({'error': 'Project with that name already exists'}), 409

    return jsonify({'message': f'Project {code} created successfully'}), 201


@REQUEST_API.route('/', methods=['GET'])
def get_projects():
    all_projects = get_all_projects()

    project_schema = Project(many=True)
    serialized_projects = project_schema.dump(all_projects)

    return serialized_projects


@REQUEST_API.route('/<int:project_id>', methods=['GET'])
def get_info(project_id):
    project_info = get_project_info(project_id)

    if project_info is None:
        return jsonify({'error': 'Project with that id does not exist'}), 404

    return jsonify(project_info)


@REQUEST_API.route('/<int:project_id>/settings', methods=['PUT'])
def change_project_settings(project_id):
    data = request.json
    errors = ProjectSettingsSchema().validate(data)

    if errors:
        return jsonify({'error': f'Please check the following fields: {errors}'}), 400

    error_code = change_settings(project_id, data)

    if error_code == 1:
        return jsonify({'error': f'Project with the id of {project_id} does not exist!'}), 404
    if error_code == 2:
        return jsonify({'error': f'Project with the id of {project_id} does not have a settings file!'}), 404

    return jsonify({'message': f'Successfully updated these settings: {data}'}), 201


@REQUEST_API.route('/<int:project_id>/models', methods=['GET'])
def get_project_models(project_id):
    project_models = get_models(project_id)

    if project_models is None:
        return jsonify({'error': 'Project with that id does not exist'}), 404

    model_schema = Model(many=True)
    serialized_models = model_schema.dump(project_models)

    return serialized_models


@REQUEST_API.route('/<int:project_id>/models/<int:model_id>/download', methods=['GET'])
def download_model(project_id, model_id):
    model = get_model(project_id, model_id)
    if not model:
        return jsonify({'error': 'Model with the following project_id was not found!'}), 404

    binary_data = model.model
    pt_data = torch.load(io.BytesIO(binary_data))
    pt_file = io.BytesIO()
    torch.save(pt_data, pt_file)
    pt_file.seek(0)
    return send_file(pt_file, mimetype='application/octet-stream', as_attachment=True,
                     download_name=f'model_{model_id}.pt')


@REQUEST_API.route('/<int:project_id>/upload', methods=["POST"])
def upload(project_id: int):
    data = request.form
    errors = Upload().validate(data)

    if errors:
        return jsonify({'error': f'Please check the following fields: {errors}'}), 400

    uploader = data["uploader_name"]
    split = data["split"]
    files = request.files.getlist("files")
    if files is None:
        return jsonify({'error': f'Files field not found'}), 400
    uploaded_files = _check_files(files)
    if type(uploaded_files) is tuple:
        return uploaded_files
    # passed, failed, annotations = upload_files(uploaded_files, project_id, uploader, split)

    return jsonify(
        {'message': f'Uploaded {passed} images and {annotations} annotations. There were {failed} failed images'}), 201

# @REQUEST_API.route('/<int:project_id>/zip-upload', methods=["POST"])
# def zip_upload(project_id: int):
#     data = request.form
#     errors = ZipUpload().validate(data)
#
#     if errors:
#         return jsonify({'error': f'Please check the following fields: {errors}'}), 400
#
#     uploader = data["uploader_name"]
#     split = data["split"]
#     file = request.files.get("file")
#     if file is None:
#         return jsonify({'error': "file field not found"}), 400
#     filename = secure_filename(file.filename)
#     if filename.endswith("tar.gz"):
#         uploaded_files = _check_zip_file(file)
#         if type(uploaded_files) is tuple:
#             return uploaded_files
#     else:
#         return jsonify({'error': f'not supported parsing {filename}'}), 405
#     print(uploaded_files)
#     # # upload file
#     # passed, failed, annotations = upload_files(uploaded_files, project_id, uploader, split)
#
#     return jsonify(
#         {'message': f'Uploaded {passed} images and {annotations} annotations. There were {failed} failed images'}), 201
