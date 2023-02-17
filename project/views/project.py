import io
import zipfile
import tarfile
from io import BytesIO

import PIL
from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from werkzeug.utils import secure_filename

from project.schemas.zip_upload import ZipUpload
from project.services.file_upload_service import upload_files
from project.models.annotation import Annotation
from project.models.image import Image
from project.services.project_service import create_project, get_models, get_all_projects, get_project_info, \
    change_settings
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


def _filter_files(files):
    for f in files:
        if f.content_type in image_types or f.content_type in text_types:
            for item in _filestorage_to_db_item(f):
                yield item
        else:
            raise ValidationError("only images and txt files can be uploaded")


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
            _list.append((Annotation(x_center=x, y_center=y, width=w, height=h, class_id=nr), name))
        except Exception as e:
            raise ValidationError(f"txt file named: {name} did not follow yolo format.")
    return _list


def _filestorage_to_db_item(f):
    content = f.stream.read()
    if f.content_type in text_types:  # text file
        text = str(content, "utf-8")
        name = secure_filename(f.filename).split(".")[0]
        _list = _text_to_annotations(text, name)
        f.stream.close()
        return _list
    else:  # image file
        io_bytes = BytesIO(content)
        img = PIL.Image.open(io_bytes)
        f.stream.close()
        name = secure_filename(f.filename).split(".")[0]
        return (Image(image=content, width=img.size[0], height=img.size[1]), name),


"""
/api/projects/43/images -> ?page=1&number=20 ##
/api/projects/43/models/3 get 1 model info +
/api/projects/43/models/3/download get download model +

/api/queue -> get +
/api/queue -> post add project ##
"""


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


@REQUEST_API.route('/<int:project_id>/upload', methods=["POST"])
def upload(project_id: int):
    data = Upload().load(request.form)
    uploader = data["uploader_name"]
    split = data["split"]
    uploaded_files = request.files.getlist("files")
    uploaded_files = [f for f in _filter_files(uploaded_files)]
    passed, failed, annotations = upload_files(uploaded_files, project_id, uploader, split)

    return jsonify(
        {'message': f'Uploaded {passed} images and {annotations} annotations. There were {failed} failed images'}), 201


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

    return jsonify({'message': f'Successfully updated these settings:{data}'}), 201


@REQUEST_API.route('/<int:project_id>/models', methods=['GET'])
def get_project_models(project_id):
    project_models = get_models(project_id)

    if project_models is None:
        return jsonify({'error': 'Project with that id does not exist'}), 404

    model_schema = Model(many=True)
    serialized_models = model_schema.dump(project_models)

    return serialized_models

def _check_zip_file(file):
    content = file.stream.read()
    file.stream.close()
    tar = tarfile.open(fileobj=BytesIO(content))
    for item in tar:
        if not item.isfile():
            continue
        _bytes = tar.extractfile(item.name).read()
        for db_object in _bytes_to_db_object(_bytes, item.name):
            yield db_object

def _bytes_to_db_object(_bytes, name: str):

    if name.endswith(".txt"):
        text = str(_bytes, "utf-8")
        name = secure_filename(name).split(".")[0]
        _list = _text_to_annotations(text, name)
        return _list
    elif name.endswith(".png") or name.endswith(".jpeg"):  # image file
        io_bytes = BytesIO(_bytes)
        img = PIL.Image.open(io_bytes)
        name = secure_filename(name).split(".")[0]
        return (Image(image=_bytes, width=img.size[0], height=img.size[1]), name),
    else:
        raise ValidationError("")



@REQUEST_API.route('/<int:project_id>/zip-upload', methods=["POST"])
def zip_upload(project_id: int):
    data = ZipUpload().load(request.form)
    uploader = data["uploader_name"]
    split = data["split"]
    file = request.files.get("file")
    if file is None:
        raise ValidationError("file field not found")
    # check file
    uploaded_files = [f for f in _check_zip_file(file)]

    # upload file
    passed, failed, annotations = upload_files(uploaded_files, project_id, uploader, split)

    return jsonify(
        {'message': f'Uploaded {passed} images and {annotations} annotations. There were {failed} failed images'}), 201