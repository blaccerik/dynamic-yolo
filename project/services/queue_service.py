import os
import subprocess

from sqlalchemy import func, asc

from project import db
from project.models.project import Project
from project.models.project_status import ProjectStatus
from project.models.queue import Queue
from project.models.task import Task


def update_queue(app):
    """
    Call this function to check for updates in the queue
    :return:
    """
    """
    RuntimeError: The server socket has failed to listen on any local network address. 
    The server socket has failed to bind to [::]:29500 (errno: 98 - Address already in use). 
    The server socket has failed to bind to 0.0.0.0:29500 (errno: 98 - Address already in use).
    """
    with app.app_context():
        queue = Queue.query.order_by(asc(Queue.position)).all()
        # check if anything is in the queue
        if len(queue) == 0:
            return

        # check if anything is training
        ps = ProjectStatus.query.filter(ProjectStatus.name.like("busy")).first()
        ps_done = ProjectStatus.query.filter(ProjectStatus.name.like("done")).first()
        ps_idle = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()
        entry = Project.query.filter(Project.project_status_id == ps.id).first()
        if entry is not None:
            return

        first = queue.pop(0)
        project_id = first.project_id

        # update project status
        project = Project.query.get(project_id)

        if project.project_status_id == ps_done:
            project.project_status_id = ps_idle
            db.session.add(project)
            db.session.commit()
            return

        # project cant be in error state
        ps_error = ProjectStatus.query.filter(ProjectStatus.name.like("error")).first()
        start = False
        if project.project_status_id != ps_error.id:
            project.project_status_id = ps.id
            db.session.add(project)
            db.session.commit()
            start = True

        # remove from queue
        db.session.delete(first)
        db.session.flush()

        # update queue
        for q in queue:
            q.position = q.position - 1
            db.session.add(q)
        db.session.commit()

        # dont train project if its in error state
        if not start:
            print("project is in error state")
            return

        # train
        subprocess.Popen([
            "python",
            "-m", "torch.distributed.run",
            "--nproc_per_node", "1",
            "training_session/main.py",
            "--project_id", str(project.id),
            "--task_id", str(first.task_id)
        ])
        # error = start_training(project, first.task_id)
        # if error:
        #     new_ps = ProjectStatus.query.filter(ProjectStatus.name.like("error")).first()
        # else:
        #     new_ps = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()
        # # set project state
        # project.project_status_id = new_ps.id
        # db.session.add(project)
        # db.session.commit()


def add_to_queue(project_id: int, task_name: str, reset_counter=True):
    """
    Add project to query
    """
    project = Project.query.get(project_id)
    if project is None:
        return "Project not found!", 1

    # search queue to see if project is already there
    # if it is then don't touch it
    entry = Queue.query.filter_by(project_id=project.id).first()
    if entry is not None:
        return "Project already in queue", 2
    position = db.session.query(func.max(Queue.position)).scalar()
    if position is None:
        position = 0

    task = Task.query.filter(Task.name.like(task_name)).first()
    if task is None:
        return "Task not found", 1

    q = Queue(position=position + 1, project_id=project.id, task_id=task.id)
    db.session.add(q)

    if reset_counter:
        db.session.refresh(project)
        project.times_auto_trained = 0
        db.session.add(project)

    db.session.commit()

    return "Added to queue successfully", 3


def fetch_queue():
    return Queue.query.all()
