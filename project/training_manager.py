import logging
import os
import shutil

import yaml
from sqlalchemy import and_

from project import db, APP_ROOT_PATH
from project.models.annotation import Annotation
from project.models.image import Image
from project.models.image_class import ImageClass
from project.models.model_status import ModelStatus
from project.models.subset import Subset
from project.models.initial_model import InitialModel
from project.models.model import Model
from project.models.model_image import ModelImage
from project.models.model_results import ModelResults
from project.models.project_status import ProjectStatus
from project.models.project import Project
from project.models.project_settings import ProjectSettings

def add_results_to_db(results, maps, model_id, subset_id, epoch=None) -> ModelResults:
    metric_precision, metric_recall, metric_map_50, metric_map_50_95, val_box_loss, val_obj_loss, val_cls_loss = results

    # # ap per class
    # for index, value in enumerate(maps):
    #     print(index, value)

    # save results
    mr = ModelResults()
    if epoch is not None:
        mr.epoch = epoch
    mr.subset_id = subset_id
    mr.model_id = model_id
    mr.metric_precision = metric_precision
    mr.metric_recall = metric_recall
    mr.metric_map_50 = metric_map_50
    mr.metric_map_50_95 = metric_map_50_95
    mr.val_box_loss = val_box_loss
    mr.val_obj_loss = val_obj_loss
    mr.val_cls_loss = val_cls_loss
    db.session.add(mr)
    db.session.commit()

    return mr

# needs to be here to avoid circular import error
from project.yolo.yolov5 import val, train, detect

class TrainSession:
    def __init__(self, project: Project,
                 project_settings: ProjectSettings,
                 name: str,
                 ss_test_id: int,
                 ss_train_id: int,
                 ps_idle_id: int):
        self.ms_train = ModelStatus.query.filter(ModelStatus.name.like("training")).first()
        self.ms_test = ModelStatus.query.filter(ModelStatus.name.like("testing")).first()
        self.ms_ready = ModelStatus.query.filter(ModelStatus.name.like("ready")).first()

        self.ss_test_id = ss_test_id
        self.ss_train_id = ss_train_id
        self.ps_idle_id = ps_idle_id

        # create new model
        self.db_model = Model(model_status_id=self.ms_test.id, project_id=project.id, epochs=project_settings.epochs)

        total = 0
        prev_model_id = project.latest_model_id
        if prev_model_id is not None:  # use prev model
            prev_model = Model.query.get(prev_model_id)
            model = prev_model.model

            if model is None: # model "exists" but weights are none
                self.new_model = True
            else:
                total = prev_model.total_epochs
                self.new_model = False
                self.db_model.parent_model_id = prev_model_id

                with open(f"{APP_ROOT_PATH}/yolo/data/weights.pt", "wb") as binary_file:
                    binary_file.write(model)
        else:
            self.new_model = True

        self.db_model.total_epochs = total + project_settings.epochs

        # update database
        db.session.add(self.db_model)
        db.session.flush()

        project.latest_model_id = self.db_model.id

        db.session.add(project)
        db.session.commit()

        self.project = project
        self.project_settings = project_settings

        self.good_images = set()

        # modify yolo backbone
        if self.new_model:
            with open(f"{APP_ROOT_PATH}/yolo/yolov5/models/{name}.yaml", "r") as stream:
                yaml_file = yaml.safe_load(stream)
                yaml_file["nc"] = self.project_settings.max_class_nr

            with open(f'{APP_ROOT_PATH}/yolo/data/backbone.yaml', 'w') as outfile:
                yaml.dump(yaml_file, outfile, default_flow_style=False)



    def load_pretest(self):
        if self.new_model:
            return

        # todo use db stream and dont write to disk
        images = Image.query.filter(Image.project_id == self.project.id)
        for image in images:
            content = image.image
            with open(f"{APP_ROOT_PATH}/yolo/data/pretest_images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)

    def pretest(self):
        if self.new_model:
            return

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # load yolo settings
        opt = detect.parse_opt(True)

        setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/weights.pt")
        setattr(opt, "source", f"{APP_ROOT_PATH}/yolo/data/pretest_images")
        setattr(opt, "nosave", True)
        setattr(opt, "save_txt", True)
        setattr(opt, "save_conf", True)
        setattr(opt, "conf_thres", self.project_settings.min_confidence_threshold)
        setattr(opt, "iou_thres", self.project_settings.min_iou_threshold)

        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/pretest_results")
        setattr(opt, "name", "yolo_test")

        # run model
        detect.main(opt)

        # read results
        for path in os.listdir(f"{APP_ROOT_PATH}/yolo/data/pretest_results/yolo_test/labels"):
            self._read_file(path)

    def _read_file(self, path):
        image_file = os.path.join(f"{APP_ROOT_PATH}/yolo/data/pretest_results/yolo_test/labels", path)
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
            "path": f"{APP_ROOT_PATH}/yolo/data",
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

        with open(f'{APP_ROOT_PATH}/yolo/data/data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_data(self):

        # get images
        # todo only select images with certain timestamp
        images = Image.query.filter(Image.project_id == self.project.id)

        # filter out "good" images
        images = [image for image in images if image.id not in self.good_images]
        print(len(images), len(self.good_images))

        for image in images:

            # skip confident images
            if image.id in self.good_images:
                continue

            if image.subset_id == self.ss_train_id:
                location = f"{APP_ROOT_PATH}/yolo/data/train"
            elif image.subset_id == self.ss_test_id:
                location = f"{APP_ROOT_PATH}/yolo/data/test"
            else:
                print(image.subset_id, self.ss_train_id, self.ss_test_id)
                raise RuntimeError("subset id not found")

            # save image
            content = image.image
            with open(f"{location}/images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)


            annotations = Annotation.query.filter_by(project_id=self.project.id, image_id=image.id)
            text = ""

            # save label
            for a in annotations:
                line = f"{a.class_id} {a.x_center} {a.y_center} {a.width} {a.height}\n"
                text += line
            with open(f"{location}/labels/{image.id}.txt", "w") as text_file:
                text_file.write(text)

            # create db entry
            mi = ModelImage(model_id=self.db_model.id, image_id=image.id)
            db.session.add(mi)
        db.session.commit()

    def train(self):

        # update database
        self.db_model.model_status_id = self.ms_train.id
        db.session.add(self.db_model)
        db.session.commit()

        # set logging to warning to see much less info at console
        # logging.getLogger("utils.general").setLevel(logging.WARNING)  # yolov5 logger
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # train model with labeled images
        opt = train.parse_opt(True)

        # change some values
        setattr(opt, "data", f"{APP_ROOT_PATH}/yolo/data/data.yaml")
        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "img", self.project_settings.img_size)
        setattr(opt, "epochs", self.project_settings.epochs)

        # setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/model")
        setattr(opt, "name", "yolo_train")

        if self.new_model:
            setattr(opt, "weights", "")
            setattr(opt, "cfg", f"{APP_ROOT_PATH}/yolo/data/backbone.yaml")
        else:
            setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/weights.pt")
            setattr(opt, "cfg", "")

        # add model id to opt
        setattr(opt, "db_model_id", self.db_model.id)
        setattr(opt, "db_train_subset_id", self.ss_train_id)

        train.main(opt)  # long process

        # read model weights
        with open(f"{APP_ROOT_PATH}/yolo/data/model/yolo_train/weights/best.pt", "rb") as f:
            content = f.read()
            self.db_model.model = content

    def test(self):

        # update database
        self.db_model.model_status_id = self.ms_test.id
        db.session.add(self.db_model)
        db.session.commit()

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # load yolo settings
        opt = val.parse_opt(True)

        setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/model/yolo_train/weights/best.pt")
        setattr(opt, "data", f"{APP_ROOT_PATH}/yolo/data/data.yaml")
        setattr(opt, "task", "test")
        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/results")
        setattr(opt, "name", "yolo_test")
        # setattr(opt, "source", "app/yolo/data/test")
        # setattr(opt, "nosave", True)
        # setattr(opt, "project", "app/yolo/data/results")
        # setattr(opt, "name", "yolo_test")
        # setattr(opt, "save_txt", True)
        # setattr(opt, "save_conf", True)

        # run model
        # todo dont save results
        settings = vars(opt)
        # settings["compute_loss"] = True
        # print(**vars(opt))
        results, maps, t = val.run(**settings)
        mr = add_results_to_db(results, maps, self.db_model.id, self.ss_test_id)
        # t holds speeds per image, [a, b, c]
        # a - init time
        # b - inference time
        # c - nms time

        # see if model needs to be retrained
        if mr.metric_map_50 < self.project_settings.minimal_map_50_threshold:
            from project.queue_manager import add_to_queue  # avoid circular import
            add_to_queue(self.project.id)

    def cleanup(self):
        # if os.path.exists("project/yolo/data"):
        #     shutil.rmtree("project/yolo/data")

        # update model
        self.db_model.model_status_id = self.ms_ready.id
        db.session.add(self.db_model)
        db.session.commit()

        # update project
        self.project.project_status_id = self.ps_idle_id
        db.session.add(self.project)
        db.session.commit()

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
    create_path(f"{APP_ROOT_PATH}/yolo/data")
    create_path(f"{APP_ROOT_PATH}/yolo/data/train")
    create_path(f"{APP_ROOT_PATH}/yolo/data/train/images")
    create_path(f"{APP_ROOT_PATH}/yolo/data/train/labels")
    create_path(f"{APP_ROOT_PATH}/yolo/data/test")
    create_path(f"{APP_ROOT_PATH}/yolo/data/test/images")
    create_path(f"{APP_ROOT_PATH}/yolo/data/test/labels")

    create_path(f"{APP_ROOT_PATH}/yolo/data/pretest_images")
    create_path(f"{APP_ROOT_PATH}/yolo/data/pretest_results")

    create_path(f"{APP_ROOT_PATH}/yolo/data/results")
    create_path(f"{APP_ROOT_PATH}/yolo/data/model")

def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def start_training(project_id: int):
    print({APP_ROOT_PATH})
    """
    Start yolo training session with the latest data
    """
    project = Project.query.get(project_id)
    if project is None:
        return "project not found"
    project_settings = ProjectSettings.query.get(project_id)
    if project_settings is None:
        return "project settings not found"

    # check if backbone file is present
    name = InitialModel.query.get(project_settings.initial_model_id).name
    if not os.path.isfile(f"{APP_ROOT_PATH}/yolo/yolov5/models/{name}.yaml"):
        return "yolo backbone yaml file not found"

    # check if project has test and train data
    ss_test = Subset.query.filter(Subset.name.like("test")).first()
    ss_train = Subset.query.filter(Subset.name.like("train")).first()
    ps_idle = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()

    test_image = Image.query.filter(and_(
        Image.project_id == project.id,
        Image.subset_id == ss_test.id
    )).first()
    train_image = Image.query.filter(and_(
        Image.project_id == project.id,
        Image.subset_id == ss_train.id
    )).first()

    # todo set minimum data amounts
    if test_image is None:
        project.project_status_id = ps_idle.id
        db.session.add(project)
        db.session.commit()
        return "test images not found"
    if train_image is None:
        project.project_status_id = ps_idle.id
        db.session.add(project)
        db.session.commit()
        return "train images not found"

    # clear dirs
    if os.path.exists(f"{APP_ROOT_PATH}/yolo/data"):
        shutil.rmtree(f"{APP_ROOT_PATH}/yolo/data")
    initialize_yolo_folders()

    ts = TrainSession(project, project_settings, name, ss_test.id, ss_train.id, ps_idle.id)
    ts.load_pretest()
    ts.pretest()
    ts.load_yaml()
    ts.load_data()
    ts.train()
    ts.test()
    ts.cleanup()

