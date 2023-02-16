from project import db
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.model_status import ModelStatus


def create_project(name: str, class_nr: int) -> int:
    """
    Create a project
    """
    p = Project.query.filter(Project.name.like(name)).first()
    if p is not None:
        return -1
    project = Project(name=name)
    db.session.add(project)
    db.session.flush()
    ps = ProjectSettings(id=project.id, max_class_nr=class_nr)
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
        model_dict['model'] = model
        serialized_models.append(model_dict)

    return serialized_models


def get_all_projects():
    """
    Get all projects
    """
    return Project.query.all()
