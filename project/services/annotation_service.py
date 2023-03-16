from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.annotator import Annotator
from project import db
from marshmallow import ValidationError


def retrieve_annotation(annotation_code):
    annotation_info = db.session.query(Annotation, Annotator). \
        join(Annotator.annotations) \
        .filter(Annotation.id == annotation_code).first()

    if not annotation_info:
        return None

    annotation_dict = {
        'id': annotation_info[0].id,
        'x_center': annotation_info[0].x_center,
        'y_center': annotation_info[0].y_center,
        'width': annotation_info[0].width,
        'height': annotation_info[0].height,
        'class_id': annotation_info[0].class_id,
        'project_id': annotation_info[0].project_id,
        'image_id': annotation_info[0].image_id,
        'annotator_name': annotation_info[1].name
    }

    return annotation_dict


def change_annotation_values(annotation_code, data):
    annotation = Annotation.query.get(annotation_code)
    if annotation is None:
        raise ValidationError({"error": f"Annotation not found"})
    for k, v in data.items():
        setattr(annotation, k, v)
    db.session.add(annotation)
    db.session.commit()
