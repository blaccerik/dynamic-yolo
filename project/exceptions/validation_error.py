from flask import Blueprint, jsonify, make_response
from marshmallow import ValidationError

validation_error = Blueprint('validation_error', __name__)


@validation_error.app_errorhandler(ValidationError)
def project_not_found_handler(error):
    return make_response(jsonify(error.messages), 400)
