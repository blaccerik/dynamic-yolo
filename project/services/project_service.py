from project import db
from project.models.annotator import Annotator
from project.models.project import Project
from project.models.project_settings import ProjectSettings


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

# def get_all():
#     """
#     Get all projects
#     """
#     return Project.query.all()