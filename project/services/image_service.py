from marshmallow import ValidationError
from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload, aliased

from project import db
from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.image import Image
from project.models.project import Project


def retrieve_image(image_code: int):
    return Image.query.filter_by(id=image_code).first()

def error_is_related(image_code: int, error_code: int):
    if error_code is None:
        return
    ae = AnnotationError.query.get(error_code)
    if ae is None:
        raise ValidationError({"error": f"error not found"})
    if ae.image_id != image_code:
        raise ValidationError({"error": f"error is not for this image"})


def image_exists(image_code: int) -> bool:
    return Image.query.get(image_code) is not None


def get_all_annotations(image_code: int):
    return Annotation.query.filter(
        Annotation.image_id == image_code,
    ).all()


def get_all_annotation_errors(image_code: int):
    return AnnotationError.query.filter(
        AnnotationError.image_id == image_code
    ).all()


def get_latest_model(image_code: int):
    i = Image.query.get(image_code)
    p_id = i.project_id
    p = Project.query.get(p_id)
    return p.latest_model_id


def get_all_human_annotations(image_code: int):
    return Annotation.query.filter(and_(
        Annotation.image_id == image_code,
        Annotation.annotator_id != None
    )).all()

def get_all_annotation_errors_(image_code: int):

    # Create aliases for the Annotation table
    a1 = aliased(Annotation)
    a2 = aliased(Annotation)

    # Perform the left join query
    query = db.session.query(AnnotationError, a1, a2). \
        join(a1, AnnotationError.model_annotation_id == a1.id, isouter=True). \
        join(a2, AnnotationError.human_annotation_id == a2.id, isouter=True). \
        filter(AnnotationError.image_id == image_code).all()

    return query

def get_errors_and_correct(image_code: int, error_id: int, show_human_annotations, show_type):

    latest_model = None
    if show_type is None:
        latest_model = -1
    elif show_type == "latest":
        latest_model = get_latest_model(image_code)

    errors = []
    for x in get_all_annotation_errors_(image_code):
        ae, am, ah = x
        if error_id is not None:
            if ae.id == error_id:
                errors.append(x)
                break
        else:
            if latest_model is not None and ae.model_id != latest_model:
                continue
            errors.append(x)

    correct = [x for x in get_all_human_annotations(image_code) if show_human_annotations]

    return errors, correct
