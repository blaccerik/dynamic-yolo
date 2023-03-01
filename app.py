from flask import send_from_directory, make_response, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from marshmallow import ValidationError

from project import create_app

app = create_app('config.DevelopmentConfig')
