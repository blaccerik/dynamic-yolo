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
    # print(text)
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

def _type_and_bytes_to_objects(filename, _bytes, values):
    if filename is None:
        return
    filename = secure_filename(filename)
    if filename.endswith(".txt"):
        annotations = _text_to_annotations(_bytes, filename)
        values.extend(annotations)
    elif filename.endswith(".png") or filename.endswith(".jpg"):
        values.append(_bytes_to_image(_bytes, filename))
    elif filename.endswith(".tar.gz"):
        values.extend(_check_zip_file(_bytes))
    else:
        raise ValidationError({'error': f'not supported parsing {filename}'})


def _get_name(text_bytes):
    text = text_bytes.decode('utf-8')
    match = re.search(r'name="([^"]+)"', text)
    if match:
        value = match.group(1)
        return value
    return None

def _get_filename(text_bytes):
    text = text_bytes.decode('utf-8')
    match = re.search(r'filename="([^"]+)"', text)
    if match:
        value = match.group(1)
        return value
    return None
def _validate_stream(flask_request):

    required = {
        "uploader_name": None,
        "split": None,
        "files": []
    }

    # get boundary between files in form
    content_type = flask_request.headers.get('Content-Type')
    _, params = cgi.parse_header(content_type)
    boundary = params.get('boundary').encode()

    # boundary in header is different from the one in stream
    # i dont know why
    # its usually 2 b"-" shorter
    boundary = b"--" + boundary
    end = boundary + b"--"

    i = 100_000
    next_lines_are_file = None
    b_list = bytearray()
    values = []
    while i:
        line = flask_request.stream.readline()
        if line.strip() == boundary:
            _type_and_bytes_to_objects(next_lines_are_file, b_list, values)

            text = request.stream.readline().strip()
            _ = request.stream.readline().strip()
            value = request.stream.readline().strip()

            name = _get_name(text)
            filename = _get_filename(text)
            # _type = _type.decode('utf-8').replace("Content-Type: ", "")
            value = value.decode('utf-8')
            next_lines_are_file = filename
            b_list = bytearray()

            if name in required:
                required[name] = value
            else:
                raise ValidationError({'error': f'unknown field: {name}'})
        elif line.strip() == end:
            _type_and_bytes_to_objects(next_lines_are_file, b_list, values)
            break
        elif line == b"":
            raise ValidationError({'error': f'cant read request'})
        elif next_lines_are_file:
            b_list.extend(line)
        i -= 1
    else:
        print("error")
    # for i in required:
    #     print(i, required[i])
    # print(values)
    #
    # print(flask_request.values)
    # print(flask_request.headers)
    # print(flask_request)

    # files = _check_zip_file(b_list)
    # print(files)

@REQUEST_API.route('/<int:project_id>/upload', methods=["POST"])
def upload(project_id: int):

    files = _validate_stream(request)
    print(files)

    # # get boundary between files in form
    # content_type = request.headers.get('Content-Type')
    # _, params = cgi.parse_header(content_type)
    # boundary = params.get('boundary').encode()




        # break
        # print(boundary)
    #     print(line)
    #     print(line == boundary)
    #     if line.strip() == boundary:
    #         break
    #     break
    # print("eee")
    # while True:
    #     line = request.stream.readline()
    #     if not line.strip():
    #         break
    #     key, value = line.decode().strip().split('; ')
    #     key = key.split('"')[1]
    #     if key == 'uploader_name':
    #         uploader_name = value.split('"')[1]
    #     elif key == 'split':
    #         split = value.split('"')[1]
    #     else:
    #         form_data.add(key, value.split('"')[1])
    # with open('/path/to/save/file', 'wb') as f:
    #     while True:
    #         line = request.stream.readline()
    #         if line.strip() == boundary:
    #             break
    #     while True:
    #         headers = {}
    #         while True:
    #             line = request.stream.readline()
    #             if not line.strip():
    #                 break
    #             key, value = line.decode().strip().split(': ')
    #             headers[key.lower()] = value
    #         if not headers:
    #             break
    #         chunk_size = int(headers.get('content-length', 0))
    #         while chunk_size > 0:
    #             chunk = request.stream.read(min(chunk_size, chunk_size))
    #             if not chunk:
    #                 break
    #             f.write(chunk)
    #             chunk_size -= len(chunk)
    return jsonify({'status': 'success'}), 200

    # print("eee")
    # f = request.files
    # print("eeer")
    # print(f)
    # for f in a:
    #     print(f)
    # data = request.form
    # print("233232")
    # errors = Upload().validate(data)
    #
    # if errors:
    #     return jsonify({'error': f'Please check the following fields: {errors}'}), 400
    #
    # uploader = data["uploader_name"]
    # split = data["split"]
    # print("000")
    # files = request.files.getlist("files")
    # print("111")
    # if files is None:
    #     return jsonify({'error': f'Files field not found'}), 400
    # uploaded_files = _check_files(files)
    # if type(uploaded_files) is tuple:
    #     return uploaded_files
    # passed, failed, annotations = upload_files(uploaded_files, project_id, uploader, split)
    return jsonify("done"), 200
    # return jsonify(
    #     {'message': f'Uploaded {passed} images and {annotations} annotations. There were {failed} failed images'}), 201


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
    models = get_models(project_id)

    return models


@REQUEST_API.route('/<int:project_id>/annotations', methods=['GET'])
def get_annotations(project_id):
    page_size = request.args.get("page_size")
    page_size = validate_page_size(page_size, 20, 100)
    page_nr = request.args.get("page_nr")
    page_nr = validate_page_nr(page_nr)

    annotations = retrieve_annotations(project_id, page_nr, page_size)
    return jsonify(annotations), 200
