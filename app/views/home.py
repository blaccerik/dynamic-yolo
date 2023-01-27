from flask import render_template

from app import app, db
from app.models.project import Project
from app.queue_manager import add_to_queue, start_training
from app.training_manager import TrainSession


@app.route('/')
def hello_world():
    return 'Hello World'


# @app.route('/hey')
# def hello_world2():
#     # link_images_and_annotations()
#     return render_template("hello.html")

@app.route('/train')
def train():
    start_training(2)
    return "done"

# @app.route('/train2')
# def train2():
#     project = Project.query.get(2)
#     ts = TrainSession(project)
#     ts.load_data()
#     ts.train()
#     return "done"


@app.route('/queue')
def queue():
    add_to_queue("project")
    return "done"
