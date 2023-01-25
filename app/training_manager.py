import logging
import os

import yaml

from app import db
from app.models.annotation import Annotation
from app.models.image import Image
from app.models.model import Model
from app.models.model_status import ModelStatus
from app.models.project import Project
from app.yolo.yolov5 import train


# def _get_model(project) -> str:
#     if project.latest_model_id is None:  # new model
#
#         m = Model(model_status_id=ms.id, latest_batch=project.latest_batch, project_id=project.id)
#         db.session.add(m)
#         db.session.flush()
#
#         project.latest_model_id = m.id
#         db.session.add(project)
#         db.session.commit()
#         return "app/yolo/yolov5/yolov5s.pt"
#     # todo use prev model and write it to disk
#     return "app/yolo/yolov5/yolov5s.pt"


class TrainSession:
    def __init__(self, project: Project):
        ms = ModelStatus.query.filter_by(name="training").first()
        self.model = Model(model_status_id=ms.id, latest_batch=project.latest_batch, project_id=project.id)
        self.project = project
        db.session.add(self.model)
        db.session.flush()

        project.latest_model_id = self.model.id
        db.session.add(project)
        db.session.commit()
        # todo use prev model and write it to disk
        self.model_path = "app/yolo/yolov5/yolov5s.pt"

    def train(self, batch_nr):
        # get images
        images = Image.query.filter(Image.project_id == self.project.id).filter(Image.batch_id >= batch_nr)

        count = 0
        max_class_id = 0
        for image in images:
            annotations = Annotation.query.filter_by(project_id=self.project.id, image_id=image.id)

            # save image
            content = image.image
            with open(f"app/yolo/datasets/data/images/{count}.png", "wb") as binary_file:
                binary_file.write(content)
            text = ""
            for annotation in annotations:
                if annotation.class_id > max_class_id:
                    max_class_id = annotation.class_id
                line = f"{annotation.class_id} {annotation.x_center} {annotation.y_center} {annotation.width} {annotation.height}\n"
                text += line
            with open(f"app/yolo/datasets/data/labels/{count}.txt", "w") as text_file:
                text_file.write(text)
            count += 1

        # create yaml file
        data = {
            "path": "../datasets/data",
            "train": "images",
            "val": "images",
            "names": {}
        }

        # todo add names for classes
        for i in range(max_class_id + 1):
            data["names"][i] = i

        with open('app/yolo/yolov5/data/data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.WARNING)

        # train model with labeled images
        opt = train.parse_opt(True)

        # change some values
        setattr(opt, "data", "app/yolo/yolov5/data/data.yaml")
        setattr(opt, "batch_size", 8)
        setattr(opt, "img", 640)
        setattr(opt, "epochs", 3)
        setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        setattr(opt, "name", "erik_test")
        setattr(opt, "weights", "")
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

        ms = ModelStatus.query.filter_by(name="ready").first()
        self.model.model_status_id = ms.id
        db.session.add(self.model)
        db.session.commit()


def start_training(project_id: int, batch_nr: int):
    """
    Start yolo training session with the latest data
    """
    project = Project.query.get(project_id)
    if project is None:
        return "project not found"
    ts = TrainSession(project)
    ts.train(batch_nr)
