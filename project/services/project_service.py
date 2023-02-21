from marshmallow import ValidationError

from project import db
from sqlalchemy import func, and_
from project.models.annotation import Annotation
from project.models.image import Image
from project.models.initial_model import InitialModel
from project.models.model import Model
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.project_status import ProjectStatus
from project.models.model_status import ModelStatus
from project.models.subset import Subset


def create_project(name: str, class_nr: int, init_model: str, img_size: int) -> int:
    """
    Create a project
    """
    p = Project.query.filter(Project.name.like(name)).first()
    if p is not None:
        return -1

    im = InitialModel.query.filter(InitialModel.name.like(init_model)).first()
    if im is None:
        raise ValidationError(f"Unknown model {init_model}", "init_model")
    project = Project(name=name)
    db.session.add(project)
    db.session.flush()
    ps = ProjectSettings(id=project.id, max_class_nr=class_nr, initial_model_id=im.id, img_size=img_size)
    db.session.add(ps)
    db.session.commit()
    return project.id


def get_models(project_code: int):
    """
    Get all models of the project
    :param project_code:
    :return:
    """
    project = Project.query.get(project_code)
    if project is None:
        return None

    models = project.models
    serialized_models = []

    # Add the model_status_name for better readability
    for model in models:
        model_dict = model.__dict__
        model_status = ModelStatus.query.get(model.model_status_id)
        model_dict['model_status_name'] = model_status.name

        result = {'model_status_name': model_dict['model_status_name'],
                  'id': model_dict['id'],
                  'added': model_dict['added']}
        serialized_models.append(result)

    return serialized_models


def get_project_info(project_code: int):
    """
    Get all the important information about the project.
    :param project_code:
    :return:
    """
    project = Project.query.get(project_code)
    if project is None:
        raise ValidationError({"error": f"Project not found"})

    project_status_name = ProjectStatus.query.get(project.project_status_id).name

    test_subset_id = Subset.query.filter_by(name='test').first().id
    train_subset_id = Subset.query.filter_by(name='train').first().id

    training_images = Image.query.filter_by(project_id=project_code, subset_id=train_subset_id).count()
    test_images = Image.query.filter_by(project_id=project_code, subset_id=test_subset_id).count()
    test_images_annotations = Annotation.query.join(Image).join(Subset).filter(and_(
        Subset.id == test_subset_id,
        Annotation.project_id == project_code
    )).count()
    training_images_annotations = Annotation.query.join(Image).join(Subset).filter(and_(
        Subset.id == train_subset_id,
        Annotation.project_id == project_code
    )).count()

    total_models_in_project = len(project.models)

    total_epochs = Model.query.get(project.latest_model_id).total_epochs

    project_info = {
        'name': project.name,
        'status': project_status_name,
        'train_images_amount': training_images,
        'train_annotations': training_images_annotations,
        'test_images_amount': test_images,
        'test_annotations': test_images_annotations,
        'amount_of_models': total_models_in_project,
        'total_epochs_trained': total_epochs
    }
    return project_info


def change_settings(project_code: int, new_settings: dict):
    # TODO add exceptions to get rid of this returning numbers situation
    # TODO maybe add a check in schema to filter these settings so that
    #  the values cant be 0 for example or bigger than 1
    project = Project.query.get(project_code)

    if project is None:
        raise ValidationError({"error":  f"Project not found"})

    project_settings = ProjectSettings.query.get(project_code)

    if project_settings is None:
        raise ValidationError({"error": f"Project settings not found"})

    for k, v in new_settings.items():
        setattr(project_settings, k, v)

    # update project if its in error state
    idle_ps = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()
    error_ps = ProjectStatus.query.filter(ProjectStatus.name.like("error")).first()
    if project.project_status_id == error_ps.id:
        project.project_status_id = idle_ps.id
        db.session.add(project)
    db.session.add(project_settings)
    db.session.commit()


def get_settings(project_code: int) -> dict:
    project = Project.query.get(project_code)

    if project is None:
        raise ValidationError({"error":  f"Project not found"})
    project_settings = ProjectSettings.query.get(project_code)

    if project_settings is None:
        raise ValidationError({"error": f"Project settings not found"})

    name = InitialModel.query.get(project_settings.initial_model_id).name

    return {
        "max_class_nr": project_settings.max_class_nr,
        "epochs": project_settings.epochs,
        "batch_size": project_settings.batch_size,
        "initial_model": name,
        "confidence_threshold": project_settings.confidence_threshold,
        "train_test_ratio": project_settings.train_test_ratio,
        "minimal_map_50_threshold": project_settings.minimal_map_50_threshold,
        "min_confidence_threshold": project_settings.min_confidence_threshold,
        "min_iou_threshold": project_settings.min_iou_threshold
    }


def get_all_projects():
    """
    Get all projects
    """
    return Project.query.all()


def get_model(project_code: int, model_code: int):
    """
    Return the project that was asked for.
    :param project_code:
    :param model_code:
    :return:
    """

    return Model.query.filter_by(project_id=project_code, id=model_code).first()
