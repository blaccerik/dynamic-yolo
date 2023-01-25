from flask import render_template

from app import app, db
from app.queue_manager import add_to_queue, start_training


@app.route('/')
def hello_world():
    return 'Hello World'


# @app.route('/hey')
# def hello_world2():
#     # link_images_and_annotations()
#     return render_template("hello.html")

@app.route('/train')
def train():
    start_training(2, 3)
    return "done"


@app.route('/queue')
def queue():
    add_to_queue("project")
    return "done"
