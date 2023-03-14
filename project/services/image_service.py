from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload

from project import db
from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationErrors
from project.models.image import Image
from project.models.project import Project


def retrieve_image(image_code: int):
    return Image.query.filter_by(id=image_code).first()


def get_annotations(image_code: int):
    return Annotation.query.filter(and_(
        Annotation.image_id == image_code,
        Annotation.annotator_id != None
    )).all()


def get_errors(image_code: int):
    return db.session.query(Annotation, AnnotationErrors).options(
        joinedload(Annotation.annotation_errors_robot),
        joinedload(Annotation.annotation_errors_human)
    ).filter(
        Annotation.image_id == image_code
    ).filter(
        (Annotation.id == AnnotationErrors.id_robot) | (Annotation.id == AnnotationErrors.id_human)
    )


def get_latest_errors(image_code: int):
    i = Image.query.get(image_code)
    p_id = i.project_id
    p = Project.query.get(p_id)
    latest_model_id = p.latest_model_id
    return db.session.query(Annotation, AnnotationErrors).options(
        joinedload(Annotation.annotation_errors_robot),
        joinedload(Annotation.annotation_errors_human)
    ).filter(
        Annotation.image_id == image_code,
        AnnotationErrors.model_id == latest_model_id
    ).filter(
        (Annotation.id == AnnotationErrors.id_robot) | (Annotation.id == AnnotationErrors.id_human)
    )
