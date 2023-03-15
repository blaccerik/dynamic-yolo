from sqlalchemy import and_, or_
from sqlalchemy.orm import joinedload, aliased

from project import db
from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.image import Image
from project.models.project import Project


def retrieve_image(image_code: int):
    return Image.query.filter_by(id=image_code).first()


def image_exists(image_code: int) -> bool:
    return Image.query.get(image_code) is not None


# def get_annotations(image_code: int):
#     return Annotation.query.filter(and_(
#         Annotation.image_id == image_code,
#         Annotation.annotator_id != None
#     )).all()
#
#
# # def get_
#
# def get_errors(image_id: int):
#     query = db.session.query(AnnotationError, Annotation).filter(and_(
#         AnnotationError.id_human == Annotation.id,
#         Annotation.image_id == image_id
#     ))
#
#     query2 = db.session.query(AnnotationError, Annotation).filter(and_(
#         AnnotationError.id_robot == Annotation.id,
#         Annotation.image_id == image_id
#     ))
#
#     # Combine the two queries with an outerjoin
#     combined_query = query.outerjoin(query2, and_(
#         AnnotationError.id_human == Annotation.id,
#         AnnotationError.id_robot == Annotation.id
#     ))
#
#     # Join the subqueries on AnnotationErrors.id
#     joined_query = combined_query.subquery().join(AnnotationError)
#
#     # Select the desired columns from the joined query
#     result = db.session.query(joined_query.c.AnnotationErrors, joined_query.c.Annotation,
#                               joined_query.c.Annotation_1).all()
#     return result
#     # return db.session.query(Annotation, AnnotationErrors).options(
#     #     joinedload(Annotation.annotation_errors_robot),
#     #     joinedload(Annotation.annotation_errors_human)
#     # ).filter(
#     #     Annotation.image_id == image_code
#     # ).filter(
#     #     (Annotation.id == AnnotationErrors.id_robot) | (Annotation.id == AnnotationErrors.id_human)
#     # )

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

# def get_latest_errors(image_code: int):
#     i = Image.query.get(image_code)
#     p_id = i.project_id
#     p = Project.query.get(p_id)
#     latest_model_id = p.latest_model_id
#     return db.session.query(Annotation, AnnotationError).options(
#         joinedload(Annotation.annotation_errors_robot),
#         joinedload(Annotation.annotation_errors_human)
#     ).filter(
#         Annotation.image_id == image_code,
#         AnnotationError.model_id == latest_model_id
#     ).filter(
#         (Annotation.id == AnnotationError.id_robot) | (Annotation.id == AnnotationError.id_human)
#     )
