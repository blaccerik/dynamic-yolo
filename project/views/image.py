import cv2
import numpy as np
from flask import Blueprint, jsonify, send_file, request

from io import BytesIO

from marshmallow import ValidationError

from project.services.image_service import retrieve_image, get_annotations, get_errors, get_latest_errors

REQUEST_API = Blueprint('images', __name__, url_prefix="/images")


def add_ano_to_image(cv2_image, a, text, dh, dw, color):
    l = int((a.x_center - a.width / 2) * dw)
    r = int((a.x_center + a.width / 2) * dw)
    t = int((a.y_center - a.height / 2) * dh)
    b = int((a.y_center + a.height / 2) * dh)
    # if a.id == 36 or a.id == 9562:
    #     print(a)
    #     print(a.x_center)
    #     print(a.y_center)
    #     print(l, r, t, b)
    cv2.rectangle(cv2_image, (l, t), (r, b), color, 1)
    cv2.putText(cv2_image, text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def add_errors(errors, show_errors, cv2_image, missing, dh, dw, color2, color3):
    for i, result in enumerate(errors):
        if len(show_errors) != 0 and i not in show_errors:
            continue
        a, ae = result
        if a.annotator_id is None:
            add_ano_to_image(cv2_image, a, f"{a.class_id} {ae.confidence:.2f} ({i})", dh, dw, color2)
        else:
            missing.add(a.id)
            add_ano_to_image(cv2_image, a, f"{a.class_id} ({i})", dh, dw, color3)

@REQUEST_API.route('/<int:image_id>', methods=['GET'])
def get_image(image_id):

    show_errors = request.args.getlist("show_errors")
    for i in range(len(show_errors)):
        try:
            nr = int(show_errors[i])
            show_errors[i] = nr
        except Exception as e:
            raise ValidationError({"error": f"Bad request"})
    show_type = request.args.get("show_type")
    if show_type not in ['latest', 'all', None]:
        raise ValidationError({"error": f"Bad request"})
    show_ano = request.args.get("show_annotations")

    image = retrieve_image(image_id)
    if image is None:
        return jsonify({'error': 'Image not found!'}), 404

    image_data = image.image

    # Decode the image using cv2.imdecode
    cv2_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    dh, dw, _ = cv2_image.shape

    color = (0, 255, 0)  # green
    color2 = (0, 0, 255)  # red
    color3 = (0, 255, 255)  # yellow
    missing = set()
    if show_type == "all":
        errors = get_errors(image_id)
        add_errors(errors, show_errors, cv2_image, missing, dh, dw, color2, color3)
    elif show_type == "latest":
        errors = get_latest_errors(image_id)
        add_errors(errors, show_errors, cv2_image, missing, dh, dw, color2, color3)
    if show_ano == "true":
        anos = get_annotations(image_id)
        for a in anos:
            if a.id in missing:
                continue
            add_ano_to_image(cv2_image, a, str(a.class_id), dh, dw, color)

    params = (cv2.IMWRITE_PNG_COMPRESSION, 0)
    s, encoded_image = cv2.imencode('.png', cv2_image, params=params)
    return send_file(BytesIO(encoded_image.tobytes()), mimetype='image/png')
