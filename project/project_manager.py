from project import db
from project.models.annotator import Annotator
from project.models.project import Project
from project.models.project_settings import ProjectSettings


def create_project(name: str, class_nr: int) -> str:
    """
    Create a project
    """
    p = Project.query.filter(Project.name.like(name)).first()
    if p is not None:
        return "error"
    project = Project(name=name)
    db.session.add(project)
    db.session.flush()
    ps = ProjectSettings(id=project.id, max_class_nr=class_nr)
    db.session.add(ps)
    db.session.commit()
    return "done"