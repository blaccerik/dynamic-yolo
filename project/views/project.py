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


def _text_to_annotations(content, filename):
    _list = []
    name = filename.split(".")[0]
    text = str(content, "utf-8")
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
                raise ValidationError({"error":  f"File: {name}.txt did not yolo format"})
            if x > 1 or x < 0:
                raise ValidationError({"error":  f"File: {name}.txt did not yolo format"})
            if y > 1 or y < 0:
                raise ValidationError({"error":  f"File: {name}.txt did not yolo format"})
            if w > 1 or w < 0:
                raise ValidationError({"error":  f"File: {name}.txt did not yolo format"})
            if h > 1 or h < 0:
                raise ValidationError({"error":  f"File: {name}.txt did not yolo format"})
            _list.append((Annotation(x_center=x, y_center=y, width=w, height=h, class_id=nr), name))
        except Exception as e:
            raise ValidationError({"error":  f"Can't read file: {name}.txt"})
            # return jsonify({'error': f"Can't read file: {name}.txt"}), 400
    return _list


def _bytes_to_image(content, filename):
    try:
        img = PIL.Image.open(io.BytesIO(content))
    except Exception:
        raise ValidationError({"error": f"Can't read file: {filename}"})
    name = filename.split(".")[0]
    return Image(image=content, width=img.size[0], height=img.size[1]), name


def _check_files(files):
    final_files = []
    for file in files:
        filename = secure_filename(file.filename)
        content = file.stream.read()
        file.stream.close()
        if filename.endswith(".txt"):
            annotations = _text_to_annotations(content, filename)
            final_files.extend(annotations)
        elif filename.endswith(".png") or filename.endswith(".jpg"):
            final_files.append(_bytes_to_image(content, filename))
        elif filename.endswith(".tar.gz"):
            final_files.extend(_check_zip_file(content))
        else:
            raise ValidationError({'error': f'not supported parsing {filename}'})
    return final_files


def _check_zip_file(content):
    try:
        tar = tarfile.open(fileobj=BytesIO(content))
    except Exception:
        return ValidationError({'error': f'tar file could not be opened'})
    files = []
    for item in tar:
        if not item.isfile():
            continue
        _bytes = tar.extractfile(item.name).read()
        filename = item.name
        if filename.endswith(".txt"):
            files.extend(_text_to_annotations(_bytes, filename))
        elif filename.endswith(".png") or filename.endswith(".jpg"):
            files.append(_bytes_to_image(_bytes, filename))
        else:
            raise ValidationError({'error': f'not supported parsing {filename}'})
    return files



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

    # check that all images have unique name
    seen = set()
    for f, name in uploaded_files:
        if type(f) is not Image:
            continue
        if name in seen:
            return jsonify({'error': f'Duplicate images found: {name}'}), 400
        seen.add(name)
    passed, failed, annotations = upload_files(uploaded_files, project_id, uploader, split)
    return jsonify(
        {'message': f'Uploaded {passed} images and {annotations} annotations. There were {failed} failed images'}), 201

