from flask import render_template, Blueprint

from project.models.project import Project
from project.queue_manager import add_to_queue, start_training
from project.training_manager import TrainSession

mod = Blueprint('home', __name__)

@mod.route('/')
def hello_world():
    return 'Hello World'


# @project.route('/hey')
# def hello_world2():
#     # link_images_and_annotations()
#     return render_template("hello.html")

@mod.route('/train')
def train():
    start_training(2)
    return "done"

@mod.route('/train2')
def train2():
    project = Project.query.get(2)
    ts = TrainSession(project)
    ts.load_data()
    ts.train()
    return "done"


@mod.route('/queue')
def queue():
    add_to_queue("project")
    return "done"
