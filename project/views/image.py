from flask import Blueprint, jsonify, send_file

from io import BytesIO
from project.services.image_service import retrieve_image

REQUEST_API = Blueprint('images', __name__, url_prefix="/images")


@REQUEST_API.route('/<int:image_id>', methods=['GET'])
def get_image(image_id):
    image = retrieve_image(image_id)

    if image is None:
        return jsonify({'error': 'Image not found!'}), 404

    image_data = image.image

    return send_file(BytesIO(image_data), mimetype='image/png')
