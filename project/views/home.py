import json
import os

import requests
import torch
from flask import Blueprint

from project import db
from project.models.model import Model
from project.models.project import Project
from project.services.queue_service import add_to_queue, start_training
from project.services.training_service import TrainSession
import docker

mod = Blueprint('home', __name__)


@mod.route('/')
def hello_world():
    return 'Hello World'



@mod.route('/hey')
def hello_world2():
    with open("model_27.pt", "br") as f:
        a = f.read()
        m = Model.query.get(1)
        m.model = a
        db.session.add(m)
        db.session.commit()
    return "eee"

@mod.route('/train')
def train():
    client = docker.from_env()
    container = client.containers.run(
        'eee:latest',
        command='python calculate.py',
        detach=True,
        remove=True
    )
    container.wait()
    for i in os.listdir():
        print(i)
    with open('/data2/result.json', 'r') as f:
        data = json.load(f)
    print(data)

    # Trigger the calculation
    # response = requests.get('http://calculation:8000/calculate')
    # print(response)
    return "done"


# @mod.route('/train2')
# def train2():
#     project = Project.query.get(2)
#     ts = TrainSession(project)
#     ts.load_data()
#     ts.train()
#     return "done"
