from project.models.annotation import Annotation
from project.models.annotator import Annotator
from project import db
from marshmallow import ValidationError


def retrieve_annotation(annotation_code):
    annotation = db.session.query(Annotation, Annotator).join(Annotator,
                                                              Annotation.annotator_id == Annotator.id).filter(
        Annotation.id == annotation_code).first()

    if not annotation:
        return None

    annotation_dict = {
        'id': annotation[0].id,
        'confidence': annotation[0].confidence,
        'x_center': annotation[0].x_center,
        'y_center': annotation[0].y_center,
        'width': annotation[0].width,
        'height': annotation[0].height,
        'class_id': annotation[0].class_id,
        'project_id': annotation[0].project_id,
        'image_id': annotation[0].image_id,
        'annotator_name': annotation[1].name
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
