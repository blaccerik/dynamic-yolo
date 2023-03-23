import cv2
import numpy as np
from flask import Blueprint, jsonify, send_file, request

from io import BytesIO

from marshmallow import ValidationError

from project.models.annotation import Annotation
from project.services.image_service import *

REQUEST_API = Blueprint('images', __name__, url_prefix="/images")


# class Join:
#     def __init__(self, ae: AnnotationError, a_human: Annotation, a_model: Annotation):
#         self.a_model = a_model
#         self.a_human = a_human
#         self.ae = ae


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


# def deal_with_annotations(image_id, show_type, show_human_annotations):
#
#     latest_model = None
#     if show_type == "latest":
#         latest_model = get_latest_model(image_id)
#
#     skip = set()
#     errors = []
#     correct = []
#
#     # get all annotations
#     anos = {a.id: a for a in get_all_annotations(image_id)}
#     not_correct = set()
#
#     # sort them
#     for ae in get_all_annotation_errors(image_id):
#         if latest_model is not None and ae.model_id != latest_model:
#             skip.add(ae.model_annotation_id)
#             continue
#         if ae.human_annotation_id is None:
#             errors.append(Join(ae, None, anos[ae.model_annotation_id]))
#         elif ae.model_annotation_id is None:
#             not_correct.add(ae.human_annotation_id)
#             errors.append(Join(ae, anos[ae.human_annotation_id], None))
#         else:
#             not_correct.add(ae.human_annotation_id)
#             errors.append(Join(ae, anos[ae.human_annotation_id], anos[ae.model_annotation_id]))
#
#     # find correct anos
#     if show_human_annotations:
#         for a in anos.values():
#             if a.id in not_correct or a.annotator_id is None:
#                 continue
#             correct.append(a)
#     return errors, correct


@REQUEST_API.route('/<int:image_id>', methods=['GET'])
def get_image(image_id):
    show_type = request.args.get("show_type")
    if show_type not in ['latest', 'all', None]:
        raise ValidationError({"error": f"Bad request"})

    show_ano = request.args.get("show_annotations")
    if show_ano == "true":
        show_ano = True
    elif show_ano == "false":
        show_ano = False
    elif show_ano is not None:
        raise ValidationError({"error": f"Bad request"})

    error_id = request.args.get("error_id")
    if error_id is not None:
        try:
            error_id = int(error_id)
        except:
            raise ValidationError({"error": f"Bad request"})

    image = retrieve_image(image_id)
    if image is None:
        return jsonify({'error': 'Image not found!'}), 404

    error_is_related(image_id, error_id)
    image_data = image.image

    # Decode the image using cv2.imdecode
    cv2_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # get related errors and human made annotations
    errors, correct = get_errors_and_correct(image_id, error_id, show_ano, show_type)

    dh, dw, _ = cv2_image.shape
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)
    color_green = (0, 255, 0)

    skip = set()
    for ae, am, ah in errors:
        if ah is None:
            text = f"{am.class_id} {ae.confidence:.2f} {am.id}"
            add_ano_to_image(cv2_image, am, text, dh, dw, color_red)
        elif am is None:
            text = f"{ah.class_id} {ah.id}"
            add_ano_to_image(cv2_image, ah, text, dh, dw, color_yellow)
        else:
            text = f"{ah.class_id} {ah.id}"
            add_ano_to_image(cv2_image, ah, text, dh, dw, color_yellow)
            text = f"{am.class_id} {ae.confidence:.2f} {am.id}"
            add_ano_to_image(cv2_image, am, text, dh, dw, color_red)
            skip.add(ah.id)

    for a in correct:
        if a.id in skip:
            continue
        text = f"{a.class_id} {a.id}"
        add_ano_to_image(cv2_image, a, text, dh, dw, color_green)

    params = (cv2.IMWRITE_PNG_COMPRESSION, 0)
    s, encoded_image = cv2.imencode('.png', cv2_image, params=params)
    return send_file(BytesIO(encoded_image.tobytes()), mimetype='image/png')


def get_dict_from_ano(ano: Annotation, ae: AnnotationError):
    return {
        "annotation_id": ano.id,
        "x_center": ano.x_center,
        "y_center": ano.y_center,
        "width": ano.width,
        "height": ano.height,
        "class_id": ano.class_id,
        "model_id": ae.model_id,
        "training_amount": ae.training_amount,
        "confidence": ae.confidence,
        "error_id": ae.id
    }


@REQUEST_API.route('/<int:image_id>/errors', methods=['GET'])
def get_image_errors(image_id):
    if not image_exists(image_id):
        return jsonify({'error': 'Image not found!'}), 404

    show_type = request.args.get("show_type")
    if show_type not in ['latest', 'all', None]:
        raise ValidationError({"error": f"Bad request"})

    error_id = request.args.get("error_id")
    if error_id is not None:
        try:
            error_id = int(error_id)
        except:
            raise ValidationError({"error": f"Bad request"})

    # get related errors and human made annotations
    errors, _ = get_errors_and_correct(image_id, error_id, False, show_type)
    missing_human_annotations = []
    missing_model_annotations = []
    mismatch = []
    for ae, am, ah in errors:
        if ah is None:
            missing_human_annotations.append(get_dict_from_ano(am, ae))
        elif am is None:
            missing_model_annotations.append(get_dict_from_ano(ah, ae))
        else:
            dict_ = {
                "human_annotation": get_dict_from_ano(ah, ae),
                "model_annotation": get_dict_from_ano(am, ae)
            }
            dict_["human_annotation"]["confidence"] = None
            mismatch.append(dict_)

    data = {
        "missing_human_annotations": missing_human_annotations,
        "missing_model_annotations": missing_model_annotations,
        "mismatch": mismatch
    }
    return jsonify(data), 200
