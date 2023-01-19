from flask import render_template

from app import app, db
from app.api import start_training


@app.route('/')
def hello_world():
    return 'Hello World'


# @app.route('/hey')
# def hello_world2():
#     # link_images_and_annotations()
#     return render_template("hello.html")

@app.route('/train')
def train():
    start_training()
    return "done"
