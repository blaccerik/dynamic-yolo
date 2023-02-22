from flask import Blueprint, jsonify

user_not_authorized_error = Blueprint('user_not_authorized_error', __name__)


class UserNotAuthorized(Exception):
    def __init__(self, message):
        self.message = message


@user_not_authorized_error.app_errorhandler(UserNotAuthorized)
def user_not_authorized_handler(e):
    response = jsonify({'error': e.message})
    response.status_code = 401
    return response
