from flask import Blueprint, request, jsonify, send_file
from project.services.model_service import get_models, get_model

import io
import torch

REQUEST_API = Blueprint('models', __name__, url_prefix="/models")


@REQUEST_API.route('/', methods=['GET'])
def get_all_models():
    models = get_models()

    return models


@REQUEST_API.route('/<int:model_id>/download', methods=['GET'])
def download_model(model_id):
    model = get_model(model_id)
    if not model:
        return jsonify({'error': 'Model was not found in the database!'}), 404
    binary_data = model.model
    if not binary_data:
        return jsonify({'error': 'Incorrect binary data of model!'}), 404

    pt_data = torch.load(io.BytesIO(binary_data))
    pt_file = io.BytesIO()
    torch.save(pt_data, pt_file)
    pt_file.seek(0)
    return send_file(pt_file, mimetype='application/octet-stream', as_attachment=True,
                     download_name=f'model_{model_id}.pt')
