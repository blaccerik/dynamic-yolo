import logging
import os
import shutil
import yaml

from project import db
from project.models.annotation import Annotation
from project.models.image import Image
from project.models.image_class import ImageClass
from project.models.model import Model
from project.models.model_image import ModelImage
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.yolo.yolov5 import detect, train


class TrainSession:
    def __init__(self, project: Project, project_settings: ProjectSettings):
        ms = ModelStatus.query.filter_by(name="training").first()
        model_id = project.latest_model_id

        # create new model
        self.model = Model(model_status_id=ms.id, project_id=project.id)
        self.model_path = "app/yolo/yolov5/yolov5s.pt"
        self.new_model = True
        if model_id is not None:
            # todo use correct model path
            self.new_model = False
            self.model.parent_model_id = model_id

        # update database
        self.project = project
        self.project_settings = project_settings
        db.session.add(self.model)
        db.session.flush()
        project.latest_model_id = self.model.id
        db.session.add(project)
        db.session.commit()

        self.good_images = set()

    def test_model(self):

        # if is new model then skip testing it
        if self.new_model:
            return

        # todo use db stream and dont write to disk
        images = Image.query.filter(Image.project_id == self.project.id)
        for image in images:
            content = image.image
            with open(f"app/yolo/data/test/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)

        # load yolo settings
        opt = detect.parse_opt(True)
        # todo use prev model
        setattr(opt, "weights", "app/yolo/data/best.pt")
        setattr(opt, "source", "app/yolo/data/test")
        setattr(opt, "nosave", True)
        setattr(opt, "project", "app/yolo/data/results")
        setattr(opt, "name", "yolo_test")
        setattr(opt, "save_txt", True)
        setattr(opt, "save_conf", True)

        # run model
        detect.main(opt)

        # get only bad images
        # if multiple directories then skip all
        if os.path.exists("app/yolo/data/results/yolo_test"):
            for path in os.listdir("app/yolo/data/results/yolo_test/labels"):
                self._read_file(path, )

        # clear images
        for i in os.listdir("app/yolo/data/test"):
            image = os.path.join("app/yolo/data/test", i)
            os.remove(image)

        # clear results
        for directory in os.listdir("app/yolo/data/results"):
            shutil.rmtree(os.path.join("app/yolo/data/results", directory))

    def _read_file(self, path):
        image_file = os.path.join("app/yolo/data/results/yolo_test/labels", path)
        with open(image_file, "r") as f:
            for line in f.readlines():
                class_nr, _, _, _, _, conf = line.strip().split(" ")
                conf = float(conf)
                if conf < self.project_settings.confidence_threshold:
                    return
            nr, _ = path.split(".")
            nr = int(nr)
            self.good_images.add(nr)

    def create_test_data(self):
        # get images
        images = Image.query.filter(Image.project_id == self.project.id)

        for image in images:
            if image.id in self.good_images:
                continue
            annotations = Annotation.query.filter_by(project_id=self.project.id, image_id=image.id)

            # save image
            content = image.image
            with open(f"app/yolo/data/images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)
            text = ""

            # save label
            for a in annotations:
                line = f"{a.class_id} {a.x_center} {a.y_center} {a.width} {a.height}\n"
                text += line
            with open(f"app/yolo/data/labels/{image.id}.txt", "w") as text_file:
                text_file.write(text)

            # create db entry
            mi = ModelImage(model_id=self.model.id, image_id=image.id)
            db.session.add(mi)
        db.session.commit()

        # create yaml file
        data = {
            "path": "../data",
            "train": "images",
            "val": "images",
            "names": {}
        }

        for i in range(self.project_settings.max_class_nr):
            image_class = ImageClass.query.get((self.project.id, i))
            if image_class is None:
                data["names"][i] = i
            else:
                data["names"][i] = image_class.name

        with open('app/yolo/data/data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def train(self):

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.WARNING)

        # train model with labeled images
        opt = train.parse_opt(True)

        # change some values
        setattr(opt, "data", "app/yolo/data/data.yaml")
        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "img", self.project_settings.img_size)
        setattr(opt, "epochs", self.project_settings.epochs)

        setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        setattr(opt, "project", "app/yolo/data/model")
        setattr(opt, "name", "yolo_test")
        # setattr(opt, "weights", "")
        # opt.__setattr__("cfg", "yolov5n6.yaml")  # use untrained model
        setattr(opt, "weights", self.model_path)  # use trained model

        train.main(opt)  # long process

        # delete images from disk
        for i in os.listdir("app/yolo/data/labels"):
            label = os.path.join("app/yolo/data/labels", i)
            os.remove(label)

        for i in os.listdir("app/yolo/data/images"):
            image = os.path.join("app/yolo/data/images", i)
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
    project_settings = ProjectSettings.query.get(project_id)
    if project_settings is None:
        return "project settings not found"
    ts = TrainSession(project, project_settings)
    ts.test_model()
    ts.create_test_data()
    ts.train()
