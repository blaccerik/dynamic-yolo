from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.annotator import Annotator
from project import db
from marshmallow import ValidationError


def retrieve_annotation(annotation_code):
    annotation_info = db.session.query(Annotation, Annotator). \
        join(Annotator.annotations) \
        .filter(Annotation.id == annotation_code).first()
    annotation = Annotation.query.get(annotation_code)
    if annotation is None:
        return None
    if annotation.annotator_id is None:
        name = None
    else:
        name = Annotator.query.get(annotation.annotator_id).name

    annotation_dict = {
        'id': annotation.id,
        'x_center': annotation.x_center,
        'y_center': annotation.y_center,
        'width': annotation.width,
        'height': annotation.height,
        'class_id': annotation.class_id,
        'project_id': annotation.project_id,
        'image_id': annotation.image_id,
        'annotator_name': name
    }

    return annotation_dict


def change_annotation_values(annotation_code, data):
    annotation = Annotation.query.get(annotation_code)
    if annotation is None:
        raise ValidationError({"error": f"Annotation not found"})
    if not data:
        raise ValidationError('There were not any settings changed!')

    for k, v in data.items():
        setattr(annotation, k, v)

    all_errors = annotation.model_annotation_errors + annotation.human_annotation_errors

    for e in all_errors:
        db.session.delete(e)

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
    user_id = Annotator.query.filter(Annotator.name == uploader_name).first().id
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


def choose_between_annotations_to_keep(annotation_error_id, user_name, keep_value):
    # TODO Fix the error statements
    possible_keep_values = ['model', 'human', 'both']
    annotation_error = AnnotationError.query.get(annotation_error_id)
    user = Annotator.query.filter(Annotator.name == user_name).first()

    if not user:
        raise ValidationError("User not found")

    if not annotation_error:
        raise ValidationError("Annotation error not found")

    if keep_value not in possible_keep_values:
        raise ValidationError("Wrong keep option")

    human_annotation = Annotation.query.get(annotation_error.human_annotation_id)
    model_annotation = Annotation.query.get(annotation_error.model_annotation_id)

    if keep_value == 'model':
        if model_annotation:
            model_annotation.annotator_id = user.id
            db.session.delete(human_annotation)
            db.session.delete(annotation_error)
        else:
            return 0, "Annotation error does not have model annotation"

    elif keep_value == 'human':
        if model_annotation:
            db.session.delete(model_annotation)
        db.session.delete(annotation_error)

    elif keep_value == 'both':
        if model_annotation:
            model_annotation.annotator_id = user.id
        db.session.delete(annotation_error)
    db.session.commit()
    return 1, "Success"
