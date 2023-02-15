from flask import Blueprint, request, jsonify

from project.schemas.user import User
from project.services.user_service import create_user

REQUEST_API = Blueprint('users', __name__, url_prefix="/users")


@REQUEST_API.route('/', methods=['POST'])
def create_record():
    """

    """
    data = User().load(request.json)
    name = data["name"]
    msg = create_user(name)
    if msg == "error":
        return jsonify({'error': 'User with that name already exists'}), 409

    return jsonify({'message': 'User created successfully'}), 201
