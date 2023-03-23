import os
import subprocess

from sqlalchemy import func, asc

from project import db
from project.models.project import Project
from project.models.project_settings import ProjectSettings
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
        ps_idle = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()
        entry = Project.query.filter(Project.project_status_id == ps.id).first()
        if entry is not None:
            return

        first = queue.pop(0)
        project_id = first.project_id

        # update project status
        project = Project.query.get(project_id)

        # if project.project_status_id == ps_done:
        #     project.project_status_id = ps_idle
        #     db.session.add(project)
        #     db.session.commit()
        #     return

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

        # get devices
        ps = ProjectSettings.query.get(project_id)
        dev_string = str(ps.devices)
        number = dev_string.count(",") + 1
        # train
        if number <= 1:
            process = subprocess.Popen([
                "python",
                "main.py",
                "--project_id", str(project.id),
                "--task_id", str(first.task_id)
            ])
        else:
            process = subprocess.Popen([
                "python",
                "-m", "torch.distributed.run",
                "--nproc_per_node", str(number),
                "main.py",
                "--project_id", str(project.id),
                "--task_id", str(first.task_id)
            ])
        exit_code = process.wait()
        print("subprocess finished")
        db.session.refresh(project)
        project_settings = ProjectSettings.query.get(project_id)
        if exit_code == 0:  # all good
            new_ps = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()

            # find if it needs to add to queue
            if project.auto_train_count > 0:
                task = ""
                if project_settings.always_check:
                    task = "check"
                task = task + "train"
                if project_settings.always_test:
                    task = task + "test"
                add_to_queue(project_id, task, reset_counter=False)
        else:
            new_ps = ProjectStatus.query.filter(ProjectStatus.name.like("error")).first()

        db.session.refresh(project)
        project.project_status_id = new_ps.id
        db.session.add(project)
        db.session.commit()


def add_to_queue(project_id: int, task_name: str, reset_counter=True):
    """
    Add project to query
    """
    project = Project.query.get(project_id)
    if project is None:
        return "Project not found!", 1
    project_settings = ProjectSettings.query.get(project_id)
    if project_settings is None:
        return "Project settings not found!", 1

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
        project.auto_train_count = project_settings.maximum_auto_train_number
        db.session.add(project)

    db.session.commit()

    return "Added to queue successfully", 3


def fetch_queue():
    return Queue.query.all()
