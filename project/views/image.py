from io import BytesIO

import cv2
import numpy as np
from flask import Blueprint, jsonify, send_file, request

from project.models.image_class import ImageClass
from project.services.image_service import *

REQUEST_API = Blueprint('images', __name__, url_prefix="/images")


def add_ano_to_image(cv2_image, a, text, dh, dw, color):
    l = int((a.x_center - a.width / 2) * dw)
    r = int((a.x_center + a.width / 2) * dw)
    t = int((a.y_center - a.height / 2) * dh)
    b = int((a.y_center + a.height / 2) * dh)
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


def gen_text(classes_dict, ano, conf):
    if ano.class_id in classes_dict:
        text = f"{classes_dict[ano.class_id]}"
    else:
        text = f"{ano.class_id}"
    text = text + f" {ano.id}"
    if conf is not None:
        text = text + f" {conf:.2f}"
    return text


@REQUEST_API.route('/<int:image_id>', methods=['GET'])
def get_image(image_id):
    model_id = request.args.get("model_id")
    if model_id not in ['latest', 'all', None]:
        try:
            model_id = int(model_id)
        except:
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
    errors, correct = get_errors_and_correct(image_id, error_id, show_ano, model_id)

    dh, dw, _ = cv2_image.shape
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)
    color_green = (0, 255, 0)

    skip = set()

    # get image classes
    classes = ImageClass.query.filter(ImageClass.project_id == image.project_id).all()
    classes_dict = {}
    for c in classes:
        if c is None:
            continue
        classes_dict[c.class_id] = c.name

    # sort errors
    for ae, am, ah in errors:
        if ah is None:
            text = gen_text(classes_dict, am, ae.confidence)
            add_ano_to_image(cv2_image, am, text, dh, dw, color_red)
        elif am is None:
            text = gen_text(classes_dict, ah, None)
            add_ano_to_image(cv2_image, ah, text, dh, dw, color_yellow)
            skip.add(ah.id)
        else:
            text = gen_text(classes_dict, ah, None)
            add_ano_to_image(cv2_image, ah, text, dh, dw, color_yellow)
            text = gen_text(classes_dict, am, ae.confidence)
            add_ano_to_image(cv2_image, am, text, dh, dw, color_red)
            skip.add(ah.id)
    for a in correct:
        if a.id in skip:
            continue
        text = gen_text(classes_dict, a, None)
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
        "image_count": ae.image_count,
        "human_annotation_count": ae.human_annotation_count,
        "confidence": ae.confidence,
        "error_id": ae.id
    }


def get_dict_from_correct_ano(ano: Annotation):
    return {
        "annotation_id": ano.id,
        "x_center": ano.x_center,
        "y_center": ano.y_center,
        "width": ano.width,
        "height": ano.height,
        "class_id": ano.class_id,
    }


@REQUEST_API.route('/<int:image_id>/json', methods=['GET'])
def get_image_errors(image_id):
    model_id = request.args.get("model_id")
    if model_id not in ['latest', 'all', None]:
        try:
            model_id = int(model_id)
        except:
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

    # get related errors and human made annotations
    errors, correct = get_errors_and_correct(image_id, error_id, show_ano, model_id)

    missing_human_annotations = []
    missing_model_annotations = []
    mismatch = []
    skip = set()
    for ae, am, ah in errors:
        if ah is None:
            missing_human_annotations.append(get_dict_from_ano(am, ae))
        elif am is None:
            missing_model_annotations.append(get_dict_from_ano(ah, ae))
            skip.add(ah.id)
        else:
            skip.add(ah.id)
            dict_ = {
                "human_annotation": get_dict_from_ano(ah, ae),
                "model_annotation": get_dict_from_ano(am, ae)
            }
            dict_["human_annotation"]["confidence"] = None
            mismatch.append(dict_)
    human_made_annotations = []
    for a in correct:
        if a.id in skip:
            continue
        human_made_annotations.append(get_dict_from_correct_ano(a))

    data = {
        "missing_human_annotations": missing_human_annotations,
        "missing_model_annotations": missing_model_annotations,
        "mismatch": mismatch,
        "human_made_annotations": human_made_annotations
    }

    return jsonify(data), 200
