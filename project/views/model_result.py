from flask import Blueprint, request, jsonify, send_file
from project.services.model_result_service import retrieve_results, retrieve_detailed_results
from project.shared.query_validators import validate_page_size, validate_page_nr

REQUEST_API = Blueprint('model_result', __name__, url_prefix="/results")


@REQUEST_API.route('/', methods=['GET'])
def get_results():
    project_id = request.args.get("project_id")
    model_id = request.args.get("model_id")
    subset_name = request.args.get("subset_name")

    page_size = request.args.get("page_size")
    page_size = validate_page_size(page_size, 20, 100)
    page_nr = request.args.get("page_nr")
    page_nr = validate_page_nr(page_nr)

    simple_results = retrieve_results(project_id, model_id, subset_name,page_size,page_nr)

    return jsonify(simple_results), 200


@REQUEST_API.route('/<int:result_id>', methods=['GET'])
def get_detailed_result(result_id):
    detailed_result = retrieve_detailed_results(result_id)
    return jsonify(detailed_result), 200
