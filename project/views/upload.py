
from io import BytesIO

from flask import render_template, request, Blueprint

from werkzeug.utils import secure_filename

from project.api import upload_files
from project.forms import UploadFileForm
from project.models.annotation import Annotation
from project.models.image import Image

import PIL.Image

REQUEST_API = Blueprint('upload',__name__)



# @REQUEST_API.route('/', methods=['POST'])
# def create_record():
#     """
#
#     """
#     data = User().load(request.json)
#     name = data["name"]
#     msg = create_user(name)
#     if msg == "error":
#         return jsonify({'error': 'User with that name already exists'}), 409
#
#     return jsonify({'message': 'User created successfully'}), 201

