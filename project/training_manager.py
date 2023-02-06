import logging
import os
import shutil
import yaml

from project import db
from project.models.annotation import Annotation
from project.models.image import Image
from project.models.image_class import ImageClass
from project.models.image_subset import ImageSubset
from project.models.model import Model
from project.models.model_image import ModelImage
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.yolo.yolov5 import val, train, detect


class TrainSession:
    def __init__(self, project: Project, project_settings: ProjectSettings):
        ms = ModelStatus.query.filter_by(name="training").first()

        # create new model
        self.db_model = Model(model_status_id=ms.id, project_id=project.id, epochs=project_settings.epochs)

        # todo read settings for start model
        # self.model_path = "project/yolo/yolov5/yolov5s.pt"

        model_id = project.latest_model_id
        total = 0
        if model_id is not None:  # use prev model
            prev_model = Model.query.get(model_id)
            model = prev_model.model

            if model is None: # model "exists" but weights are none
                self.new_model = True
            else:
                total = prev_model.total_epochs
                self.new_model = False
                self.db_model.parent_model_id = model_id

                with open(f"project/yolo/data/weights.pt", "wb") as binary_file:
                    binary_file.write(model)
                self.model_path = "project/yolo/data/weights.pt"
        else:
            self.new_model = True

        self.db_model.total_epochs = total + project_settings.epochs

        # update database
        self.project = project
        self.project_settings = project_settings

        db.session.add(self.db_model)
        db.session.flush()

        project.latest_model_id = self.db_model.id
        db.session.add(project)
        db.session.commit()

        self.good_images = set()

        print(self.new_model)


    def load_pretest(self):
        if self.new_model:
            return

        # todo use db stream and dont write to disk
        images = Image.query.filter(Image.project_id == self.project.id)
        for image in images:
            content = image.image
            with open(f"project/yolo/data/pretest_images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)

    def pretest(self):
        if self.new_model:
            return

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.WARNING)

        # load yolo settings
        opt = detect.parse_opt(True)

        setattr(opt, "weights", "project/yolo/data/weights.pt")
        setattr(opt, "source", "project/yolo/data/pretest_images")
        setattr(opt, "nosave", True)
        setattr(opt, "save_txt", True)
        setattr(opt, "save_conf", True)
        setattr(opt, "conf_thres", self.project_settings.min_confidence_threshold)
        setattr(opt, "iou_thres", self.project_settings.min_iou_threshold)

        setattr(opt, "project", "project/yolo/data/pretest_results")
        setattr(opt, "name", "yolo_test")

        # run model
        detect.main(opt)

        # read results
        for path in os.listdir("project/yolo/data/pretest_results/yolo_test/labels"):
            self._read_file(path)

    def _read_file(self, path):
        image_file = os.path.join("project/yolo/data/pretest_results/yolo_test/labels", path)
        with open(image_file, "r") as f:
            for line in f.readlines():
                class_nr, _, _, _, _, conf = line.strip().split(" ")
                conf = float(conf)
                if conf < self.project_settings.confidence_threshold:
                    return
            nr, _ = path.split(".")
            nr = int(nr)
            self.good_images.add(nr)

    def load_yaml(self):
        # create yaml file
        data = {
            "path": "../data",
            "train": "train",
            "val": "train",
            "test": "test",
            "names": {}
        }

        # if possible add names to
        for i in range(self.project_settings.max_class_nr):
            image_class = ImageClass.query.get((self.project.id, i))
            if image_class is None:
                data["names"][i] = i
            else:
                data["names"][i] = image_class.name

        with open('project/yolo/data/data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_data(self):
        db_test_subset = ImageSubset.query.filter_by(name="test").first()
        db_train_subset = ImageSubset.query.filter_by(name="train").first()

        train_test_ratio = self.project_settings.train_test_ratio

        # get images
        images = Image.query.filter(Image.project_id == self.project.id)

        # filter out "good" images
        images = [image for image in images if image.id not in self.good_images]
        print(len(images), len(self.good_images))
        count = 0
        threshold = train_test_ratio * len(images)

        for image in images:

            count += 1
            if count > threshold:
                location = "project/yolo/data/test"
                subset_id = db_test_subset.id
            else:
                location = "project/yolo/data/train"
                subset_id = db_train_subset.id

            annotations = Annotation.query.filter_by(project_id=self.project.id, image_id=image.id)

            # save image
            content = image.image
            with open(f"{location}/images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)
            text = ""

            # save label
            for a in annotations:
                line = f"{a.class_id} {a.x_center} {a.y_center} {a.width} {a.height}\n"
                text += line
            with open(f"{location}/labels/{image.id}.txt", "w") as text_file:
                text_file.write(text)

            # create db entry
            mi = ModelImage(model_id=self.db_model.id, image_id=image.id, image_subset_id=subset_id)
            db.session.add(mi)
        db.session.commit()


    def train(self):

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.WARNING)

        # train model with labeled images
        opt = train.parse_opt(True)

        # change some values
        setattr(opt, "data", "project/yolo/data/data.yaml")
        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "img", self.project_settings.img_size)
        setattr(opt, "epochs", self.project_settings.epochs)

        setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        setattr(opt, "project", "project/yolo/data/model")
        setattr(opt, "name", "yolo_train")

        if self.new_model:
            # todo read settings for number of classes
            #  and edit yaml file
            setattr(opt, "weights", "")
            setattr(opt, "cfg", "project/yolo/yolov5/models/yolov5s.yaml")
        else:
            setattr(opt, "weights", self.model_path)
            setattr(opt, "cfg", "")

        train.main(opt)  # long process

        # read model weights
        with open("project/yolo/data/model/yolo_train/weights/best.pt", "rb") as f:
            content = f.read()
            self.db_model.model = content


        # update database
        ms = ModelStatus.query.filter_by(name="ready").first()
        self.db_model.model_status_id = ms.id
        db.session.add(self.db_model)
        db.session.commit()

    def test(self):

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.WARNING)

        # load yolo settings
        opt = val.parse_opt(True)

        # todo use prev model
        setattr(opt, "weights", "project/yolo/data/model/yolo_train/weights/best.pt")
        setattr(opt, "data", "project/yolo/data/data.yaml")
        setattr(opt, "task", "test")
        setattr(opt, "project", "project/yolo/data/results")
        setattr(opt, "name", "yolo_test")
        # setattr(opt, "source", "app/yolo/data/test")
        # setattr(opt, "nosave", True)
        # setattr(opt, "project", "app/yolo/data/results")
        # setattr(opt, "name", "yolo_test")
        # setattr(opt, "save_txt", True)
        # setattr(opt, "save_conf", True)

        # run model
        val.main(opt)

        # read results

    def cleanup(self):
        shutil.rmtree("project/yolo/data")

def initialize_yolo_folders():
    """
    yolo:
      data:
        train:
          images
          labels
        test:
          images
          labels
        results
        pretest
        model
    """
    create_path("project/yolo/data")
    create_path("project/yolo/data/train")
    create_path("project/yolo/data/train/images")
    create_path("project/yolo/data/train/labels")
    create_path("project/yolo/data/test")
    create_path("project/yolo/data/test/images")
    create_path("project/yolo/data/test/labels")

    create_path("project/yolo/data/pretest_images")
    create_path("project/yolo/data/pretest_results")

    create_path("project/yolo/data/results")
    create_path("project/yolo/data/model")

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


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

    # clear dirs
    if os.path.exists("project/yolo/data"):
        shutil.rmtree("project/yolo/data")
    initialize_yolo_folders()

    ts = TrainSession(project, project_settings)

    ts.load_pretest()
    ts.pretest()
    ts.load_yaml()
    ts.load_data()
    ts.train()
    ts.test()
    ts.cleanup()
