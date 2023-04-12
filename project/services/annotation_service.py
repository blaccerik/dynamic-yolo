from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.annotator import Annotator
from project import db
from marshmallow import ValidationError

from project.models.image import Image
from project.models.project import Project
from project.models.project_settings import ProjectSettings


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
    if annotation is None:
        raise ValidationError({"error": f"Annotation not found!"})
    if annotation.annotator_id is None:
        errors = AnnotationError.query.filter(AnnotationError.model_annotation_id == annotation.id).all()
    else:
        errors = AnnotationError.query.filter(AnnotationError.human_annotation_id == annotation.id).all()
    for e in errors:
        db.session.delete(e)
    # if is human ano then anos remove related model anos
    if annotation.annotator_id is not None:
        for e in errors:
            if e.model_annotation_id is not None:
                db.session.delete(Annotation.query.get(e.model_annotation_id))
    db.session.delete(annotation)
    db.session.commit()


# def remove_annotation_and_extras(annotation_code):
#     annotation = Annotation.query.get(annotation_code)
#     if annotation is None:
#         raise ValidationError({"error": f"Annotation not found"})
#
#     extras_deleted = delete_extra_information(annotation_code)
#     #
#     # db.session.delete(annotation)
#     # db.session.commit()
#
#     return extras_deleted


def create_annotation(data):
    name = data["uploader"]
    user = Annotator.query.filter(Annotator.name == name).first()
    if user is None:
        raise ValidationError({"error": f"User not found"})
    image = Image.query.get(data["image_id"])
    if image is None:
        raise ValidationError({"error": f"Image not found"})
    project = Project.query.get(image.project_id)
    if project is None:
        raise ValidationError({"error": f"Project not found"})
    project_settings = ProjectSettings.query.get(project.id)
    if project_settings is None:
        raise ValidationError({"error": f"Project settings not found"})
    max_class_nr = project_settings.max_class_nr
    if data["class_id"] <= 0 or data["class_id"] > max_class_nr - 1:
        raise ValidationError(f"Class ID must be between 0 and {max_class_nr - 1}","class_id")
    annotation_to_upload = Annotation(x_center=data['x_center'],
                                      y_center=data['y_center'],
                                      width=data['width'],
                                      height=data['height'],
                                      class_id=data['class_id'],
                                      project_id=project.id,
                                      image_id=image.id,
                                      annotator_id=user.id,
                                      )
    db.session.add(annotation_to_upload)
    db.session.commit()
    return annotation_to_upload.id


def choose_between_annotations_to_keep(annotation_error_id, user_name, keep_value):
    annotation_error = AnnotationError.query.get(annotation_error_id)
    user = Annotator.query.filter(Annotator.name == user_name).first()

    if not user:
        raise ValidationError("User not found")

    if not annotation_error:
        raise ValidationError("Annotation error not found")

    model_ano_id = annotation_error.model_annotation_id
    human_ano_id = annotation_error.human_annotation_id
    if keep_value == 'model':
        if model_ano_id is None:
            raise ValidationError("Annotation error does not have model annotation id")

        # update model ano
        model_ano = Annotation.query.get(model_ano_id)
        model_ano.annotator_id = user.id
        db.session.add(model_ano)

        # delete error
        db.session.delete(annotation_error)
        db.session.commit()

        if human_ano_id is not None:
            # remove all traces of human ano
            delete_extra_information(human_ano_id)
    elif keep_value == "human":
        if human_ano_id is None:
            raise ValidationError("Annotation error does not have human annotation id")
        human_ano = Annotation.query.get(human_ano_id)
        errors = AnnotationError.query.filter(AnnotationError.human_annotation_id == human_ano.id).all()
        for e in errors:
            if e.model_annotation_id is not None:
                m_ano = Annotation.query.get(e.model_annotation_id)
                db.session.delete(m_ano)
            db.session.delete(e)
        db.session.commit()
    elif keep_value == "both":
        if model_ano_id is None:
            raise ValidationError("Annotation error does not have model annotation id")
        if human_ano_id is None:
            raise ValidationError("Annotation error does not have human annotation id")

        # add model ano
        # update model ano
        model_ano = Annotation.query.get(model_ano_id)
        model_ano.annotator_id = user.id
        db.session.add(model_ano)

        # delete error
        db.session.delete(annotation_error)
        db.session.commit()

        # delete other errors
        human_ano = Annotation.query.get(human_ano_id)
        errors = AnnotationError.query.filter(AnnotationError.human_annotation_id == human_ano.id).all()
        for e in errors:
            if e.model_annotation_id is not None:
                m_ano = Annotation.query.get(e.model_annotation_id)
                db.session.delete(m_ano)
            db.session.delete(e)
        db.session.commit()
    else:
        if model_ano_id is not None:
            delete_extra_information(model_ano_id)
        if human_ano_id is not None:
            delete_extra_information(human_ano_id)

    db.session.commit()
