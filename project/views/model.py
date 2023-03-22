from flask import Blueprint, request, jsonify, send_file

from project.services.model_service import get_model, model_info, upload_new_model
import io
import torch

from yolov5.models.experimental import attempt_load

REQUEST_API = Blueprint('models', __name__, url_prefix="/models")


@REQUEST_API.route('/<int:model_id>', methods=['GET'])
def get_model_info(model_id):
    data = model_info(model_id)
    return jsonify(data), 200


@REQUEST_API.route('/<int:model_id>/download', methods=['GET'])
def download_model(model_id):
    model = get_model(model_id)
    binary_data = model.model

    # this needs to be here or else torch cant load model
    attempt_load("e", binary_weights=binary_data)
    pt_data = torch.load(io.BytesIO(binary_data))
    pt_file = io.BytesIO()
    torch.save(pt_data, pt_file)
    pt_file.seek(0)
    return send_file(pt_file, mimetype='application/octet-stream', as_attachment=True,
                     download_name=f'model_{model_id}.pt')


@REQUEST_API.route('/<int:project_id>', methods=['POST'])
def upload_model(project_id):
    model_file = request.files['model_file']
    upload_new_model(project_id, model_file)

    return jsonify({'message': f'New model uploaded successfully!'}), 201
