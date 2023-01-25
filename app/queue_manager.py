from sqlalchemy import func, asc

from app import db, app
from app.models.project import Project
from app.models.queue import Queue
from app.training_manager import start_training


def update_queue():
    """
    Update queue
    :return:
    """
    with app.app_context():
        queue = Queue.query.order_by(asc(Queue.position)).all()

        # check if anything is in the queue
        if len(queue) == 0:
            return
        first = queue.pop(0)
        project_id = first.project_id

        project = Project.query.get(project_id)
        model_id = project.latest_model_id
        # no models found, create new one
        if model_id is None:
            start_training(project_id, 0)

        # update queue
        db.session.delete(first)
        db.session.commit()
        for q in queue:
            q.position = q.position - 1
            db.session.add(q)
        db.session.commit()



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
