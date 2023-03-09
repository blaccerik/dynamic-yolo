from flask import Blueprint, request, jsonify

from project.schemas.queue import QueueSchema
from project.services.queue_service import fetch_queue, add_to_queue

REQUEST_API = Blueprint('queue', __name__, url_prefix="/queue")


@REQUEST_API.route('/', methods=['GET'])
def get_queue():
    current_queue = fetch_queue()

    queue_schema = QueueSchema(many=True)
    serialized_projects = queue_schema.dump(current_queue)

    return serialized_projects


@REQUEST_API.route('/', methods=['POST'])
def add_project_to_queue():
    project_id = request.json.get('project_id')
    message, error = add_to_queue(project_id, "retrain")

    if error == 1:
        return jsonify({'error': message}), 404
    if error == 2:
        return jsonify({'error': message}), 404

    return jsonify({'message': 'Added project to queue'}), 201
