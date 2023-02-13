import time

from sqlalchemy import func, asc, or_

from project import db
from project.models.model import Model
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.queue import Queue
from project.training_manager import start_training


def update_queue(app):
    """
    Call this function to check for updates in the queue
    :return:
    """
    with app.app_context():
        queue = Queue.query.order_by(asc(Queue.position)).all()
        # check if anything is in the queue
        if len(queue) == 0:
            return
        first = queue.pop(0)
        project_id = first.project_id

        ms1 = ModelStatus.query.filter(ModelStatus.name.like("training")).first()
        ms2 = ModelStatus.query.filter(ModelStatus.name.like("testing")).first()

        # check if anything is training
        m = Model.query.filter(
            or_(
                Model.model_status_id == ms1.id,
                Model.model_status_id == ms2.id,
            )
        ).first()
        # something is training/testing
        if m is not None:
            return

        db.session.delete(first)
        db.session.flush()

        # update queue
        for q in queue:
            q.position = q.position - 1
            db.session.add(q)
        db.session.commit()

        # train
        start_training(project_id)




def add_to_queue(project_id: int):
    """
    Add project to query
    """
    project = Project.query.get(project_id)
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
