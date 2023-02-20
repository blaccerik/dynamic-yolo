from flask import Blueprint, jsonify
project_not_found_error = Blueprint('error', __name__)


class ProjectNotFoundException(Exception):
    def __init__(self, message):
        self.message = message


@project_not_found_error.errorhandler(ProjectNotFoundException)
def project_not_found_handler(e):
    response = jsonify({'message': e.message})
    response.status_code = 400
    return response
