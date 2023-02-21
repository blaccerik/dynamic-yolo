from flask import Blueprint, request, jsonify

from project.schemas.queue import QueueSchema
from project.services.queue_service import fetch_queue

REQUEST_API = Blueprint('queue', __name__, url_prefix="/queue")


@REQUEST_API.route('/', methods=['GET'])
def get_queue():
    current_queue = fetch_queue()

    queue_schema = QueueSchema(many=True)
    serialized_projects = queue_schema.dump(current_queue)

    return serialized_projects
