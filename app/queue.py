from sqlalchemy import func

from app import db
from app.models.project import Project
from app.models.queue import Queue


def add_to_queue(project_name: str):
    """
    Add project to query
    :param project_name:
    :return:
    """
    project = Project.query.filter_by(name=project_name).first()
    if project is None:
        return "project not found"

    # search queue to see if project is already there
    # if it is then don't touch it
    entry = Queue.query.filter_by(project_id=project.id).first()
    if entry is None:
        position = db.session.query(func.max(Queue.position)).scalar()
        if position is None:
            position = 0
        q = Queue(position=position + 1, project_id=project.id)
        db.session.add(q)
        db.session.commit()
