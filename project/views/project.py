import cgi
import io
import re
import tarfile
from io import BytesIO

import PIL
from flask import Blueprint, request, jsonify
from marshmallow import ValidationError
from werkzeug.datastructures import MultiDict
from werkzeug.utils import secure_filename

from project.services.file_upload_service import upload_files
from project.models.annotation import Annotation
from project.models.image import Image
from project.services.project_service import create_project, get_all_projects, get_project_info, \
    change_settings, get_settings, get_images, get_models, retrieve_annotations
from project.schemas.project import Project
from project.schemas.upload import Upload
from project.schemas.projectsettings import ProjectSettingsSchema
from project.shared.query_validators import validate_page_size, validate_page_nr

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
    if text == "":
        return [(None, name)]

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
                raise ValidationError({"error": f"File: {name}.txt did not yolo format"})
            if x > 1 or x < 0:
                raise ValidationError({"error": f"File: {name}.txt did not yolo format"})
            if y > 1 or y < 0:
                raise ValidationError({"error": f"File: {name}.txt did not yolo format"})
            if w > 1 or w < 0:
                raise ValidationError({"error": f"File: {name}.txt did not yolo format"})
            if h > 1 or h < 0:
                raise ValidationError({"error": f"File: {name}.txt did not yolo format"})
            _list.append((Annotation(x_center=x, y_center=y, width=w, height=h, class_id=nr), name))
        except Exception as e:
            raise ValidationError({"error": f"Can't read file: {name}.txt"})
            # return jsonify({'error': f"Can't read file: {name}.txt"}), 400
    return _list


def _bytes_to_image(content, filename):
    try:
        img = PIL.Image.open(io.BytesIO(content))
    except Exception:
        raise ValidationError({"error": f"Can't read file: {filename}"})
    name = filename.split(".")[0]
    return Image(image=content, width=img.size[0], height=img.size[1]), name


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


@REQUEST_API.route('/', methods=['POST'])
def create_record():
    data = Project().load(request.json)
    name = data["name"]
    max_class_nr = data["max_class_nr"]
    init_model = data["initial_model"]
    img_size = data["img_size"]
    code = create_project(name, max_class_nr, init_model, img_size)
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
    passed, failed, annotations = upload_files(uploaded_files, project_id, uploader, split)

    return jsonify(
        {'message': f'Uploaded {passed} images and {annotations} annotations. There were {failed} failed images'}), 201


@REQUEST_API.route('/<int:project_id>', methods=['GET'])
def get_info(project_id):
    project_info = get_project_info(project_id)

    return jsonify(project_info)


@REQUEST_API.route('/<int:project_id>/settings', methods=['PUT'])
def change_project_settings(project_id):
    data = request.json
    errors = ProjectSettingsSchema().validate(data)

    if errors:
        return jsonify({'error': f'Please check the following fields: {errors}'}), 400

    change_settings(project_id, data)

    return jsonify({'message': f'Successfully updated these settings: {data}'}), 201


@REQUEST_API.route('/<int:project_id>/settings', methods=['GET'])
def get_project_settings(project_id):
    settings = get_settings(project_id)
    return jsonify(settings), 201


@REQUEST_API.route('/<int:project_id>/images', methods=['GET'])
def get_project_images(project_id):
    page_size = request.args.get("page_size")
    page_size = validate_page_size(page_size, 20, 100)
    page_nr = request.args.get("page_nr")
    page_nr = validate_page_nr(page_nr)

    images = get_images(project_id, page_size, page_nr)
    return jsonify(images), 200


@REQUEST_API.route('/<int:project_id>/models', methods=['GET'])
def get_all_models(project_id):
    page_size = request.args.get("page_size")
    page_size = validate_page_size(page_size, 20, 100)
    page_nr = request.args.get("page_nr")
    page_nr = validate_page_nr(page_nr)

    models = get_models(project_id, page_nr, page_size)

    return models


@REQUEST_API.route('/<int:project_id>/annotations', methods=['GET'])
def get_annotations(project_id):
    page_size = request.args.get("page_size")
    page_size = validate_page_size(page_size, 20, 100)
    page_nr = request.args.get("page_nr")
    page_nr = validate_page_nr(page_nr)

    annotations = retrieve_annotations(project_id, page_nr, page_size)
    return jsonify(annotations), 200
