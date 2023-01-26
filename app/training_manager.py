import logging
import os
from io import BytesIO

import PIL
import torch
import yaml

from app import db
from app.models.annotation import Annotation
from app.models.image import Image
from app.models.model import Model
from app.models.model_status import ModelStatus
from app.models.project import Project
from app.yolo.yolov5 import train, detect


class TrainSession:
    def __init__(self, project: Project):
        ms = ModelStatus.query.filter_by(name="training").first()
        model_id = project.latest_model_id

        # create new model
        self.model = Model(model_status_id=ms.id, project_id=project.id)
        self.model_path = "app/yolo/yolov5/yolov5s.pt"
        if model_id is not None:
            # todo use correct model path
            self.model.parent_model_id = model_id

        # update database
        self.project = project
        db.session.add(self.model)
        db.session.flush()
        project.latest_model_id = self.model.id
        db.session.add(project)
        db.session.commit()

    def test_model(self) -> set:

        # todo use db stream and dont write to disk
        images = Image.query.filter(Image.project_id == self.project.id)
        for image in images:
            content = image.image
            with open(f"app/yolo/datasets/data/test/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)

        # load settings
        opt = detect.parse_opt(True)
        # todo use prev model
        setattr(opt, "weights", "app/yolo/datasets/data/model/best.pt")
        setattr(opt, "source", "app/yolo/datasets/data/test")
        setattr(opt, "nosave", True)
        setattr(opt, "project", "app/yolo/datasets/data/results")
        setattr(opt, "save_txt", True)
        setattr(opt, "save_conf", True)

        # run model
        detect.main(opt)

        # todo read results
        # get only bad images
        return set()

    def create_test_data(self, good_images: set):
        # get images
        images = Image.query.filter(Image.project_id == self.project.id)

        for image in images:
            if image.id in good_images:
                continue
            annotations = Annotation.query.filter_by(project_id=self.project.id, image_id=image.id)

            # save image
            content = image.image
            with open(f"app/yolo/datasets/data/images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)
            text = ""
            for a in annotations:
                line = f"{a.class_id} {a.x_center} {a.y_center} {a.width} {a.height}\n"
                text += line
            with open(f"app/yolo/datasets/data/labels/{image.id}.txt", "w") as text_file:
                text_file.write(text)

        # create yaml file
        data = {
            "path": "../datasets/data",
            "train": "images",
            "val": "images",
            "names": {}
        }

        # todo add names for classes
        for i in range(80):
            data["names"][i] = i

        with open('app/yolo/yolov5/data/data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def train(self):

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.WARNING)

        # train model with labeled images
        opt = train.parse_opt(True)

        # change some values
        setattr(opt, "data", "app/yolo/yolov5/data/data.yaml")
        setattr(opt, "batch_size", 8)
        setattr(opt, "img", 640)
        setattr(opt, "epochs", 2)
        setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        setattr(opt, "name", "erik_test")
        # setattr(opt, "weights", "")
        # opt.__setattr__("cfg", "yolov5n6.yaml")  # use untrained model
        setattr(opt, "weights", self.model_path)  # use trained model

        train.main(opt)  # long process

        # delete images from disk
        for i in os.listdir("app/yolo/datasets/data/labels"):
            label = os.path.join("app/yolo/datasets/data/labels", i)
            os.remove(label)

        for i in os.listdir("app/yolo/datasets/data/images"):
            image = os.path.join("app/yolo/datasets/data/images", i)
            os.remove(image)

        # update database
        ms = ModelStatus.query.filter_by(name="ready").first()
        self.model.model_status_id = ms.id
        db.session.add(self.model)
        db.session.commit()


def start_training(project_id: int):
    """
    Start yolo training session with the latest data
    """
    project = Project.query.get(project_id)
    if project is None:
        return "project not found"
    ts = TrainSession(project)
    good_image_ids = ts.test_model()
    ts.create_test_data(good_image_ids)
    ts.train()
