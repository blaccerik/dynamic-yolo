from flask import Blueprint, request, jsonify, send_file
from project.services.model_service import  get_model, model_info

import io
import torch

REQUEST_API = Blueprint('models', __name__, url_prefix="/models")


@REQUEST_API.route('/<int:model_id>', methods=['GET'])
def get_model_info(model_id):
    data = model_info(model_id)
    return jsonify(data), 200


@REQUEST_API.route('/<int:model_id>/download', methods=['GET'])
def download_model(model_id):
    model = get_model(model_id)
    binary_data = model.model
    pt_data = torch.load(io.BytesIO(binary_data))
    pt_file = io.BytesIO()
    torch.save(pt_data, pt_file)
    pt_file.seek(0)
    return send_file(pt_file, mimetype='application/octet-stream', as_attachment=True,
                     download_name=f'model_{model_id}.pt')
