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


def delete_extra_information(annotation_code):
    annotation = Annotation.query.get(annotation_code)
    if annotation:
        errors = annotation.model_annotation_errors + annotation.human_annotation_errors
        total_errors = len(errors)
        human_errors = len(annotation.human_annotation_errors)
        model_errors = len(annotation.model_annotation_errors)

        for error in errors:
            db.session.delete(error)
        db.session.commit()
    else:
        raise ValidationError({"error": f"Annotation not found!"})
    return total_errors, model_errors, human_errors


def remove_annotation_and_extras(annotation_code):
    annotation = Annotation.query.get(annotation_code)
    if annotation is None:
        raise ValidationError({"error": f"Annotation not found"})

    extras_deleted = delete_extra_information(annotation_code)

    db.session.delete(annotation)
    db.session.commit()

    return extras_deleted


def create_annotation(data, uploader_name):
    user_id = Annotator.query.filter(Annotator.name.like(uploader_name)).first().id
    annotation_to_upload = Annotation(x_center=data['x_center'],
                                      y_center=data['y_center'],
                                      width=data['width'],
                                      height=data['height'],
                                      class_id=data['class_id'],
                                      project_id=data['project_id'],
                                      image_id=data['image_id'],
                                      annotator_id=user_id,
                                      )
    db.session.add(annotation_to_upload)
    db.session.commit()
    return annotation_to_upload.id
