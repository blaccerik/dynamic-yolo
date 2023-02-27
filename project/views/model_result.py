from flask import Blueprint, request, jsonify, send_file
from project.services.model_result_service import retrieve_results, retrieve_detailed_results

REQUEST_API = Blueprint('model_result', __name__, url_prefix="/results")


@REQUEST_API.route('/', methods=['GET'])
def get_results():
    project_id = request.args.get("project_id")
    simple_results = retrieve_results(project_id)

    return jsonify(simple_results), 200


@REQUEST_API.route('/<int:result_id>', methods=['GET'])
def get_detailed_result(result_id):
    detailed_result = retrieve_detailed_results(result_id)
    if not detailed_result:
        return jsonify({'error': f'Result with the ID of {result_id} was not found!'}), 404

    return jsonify(detailed_result), 200
