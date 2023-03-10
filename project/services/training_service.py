import io
import logging
import math
import os
import shutil

import cv2
import numpy as np
import torch
import yaml
from sqlalchemy import and_

from project import db, APP_ROOT_PATH, DB_READ_BATCH_SIZE, NUMBER_OF_YOLO_WORKERS
from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationExtra
from project.models.image import Image
from project.models.image_class import ImageClass
from project.models.initial_model import InitialModel
from project.models.model import Model
from project.models.model_class_results import ModelClassResult
from project.models.model_image import ModelImage
from project.models.model_results import ModelResults
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.subset import Subset
from project.yolo.yolov5.utils.augmentations import letterbox
from project.yolo.yolov5.utils.general import scale_boxes, xyxy2xywh

class SqlStream:

    def __init__(self, project_id, project_settings, ss_train_id, ss_test_id, db_model):
        self.project_id = project_id
        self.project_settings = project_settings
        self.ss_train_id = ss_train_id
        self.ss_test_id = ss_test_id
        self.db_model = db_model

        self.best = None
        self.good_images = set()

        # yolo dataloader
        self.img_size = [640, 640]
        self.stride = 32
        self.auto = True


    def save_best(self, ckpt):

        buffer = io.BytesIO()
        torch.save(ckpt, buffer)
        self.best = buffer.getvalue()

    def get_all_train_images(self):
        images = Image.query.filter(and_(
            Image.project_id == self.project_id,
            Image.subset_id == self.ss_train_id
        ))
        c = 0
        for image in images.yield_per(DB_READ_BATCH_SIZE):
            content = image.image
            nparr = np.frombuffer(content, np.uint8)

            # Decode numpy array into image
            im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            yield im, im0, image.id
            c += 1
            if c == 10000:
                break

    def add_results_to_db(self, results, maps, subset, epoch=None):
        if subset == "train":
            subset_id = self.ss_train_id
        elif subset == "test":
            subset_id = self.ss_test_id
        metric_precision, metric_recall, metric_map_50, metric_map_50_95, val_box_loss, val_obj_loss, val_cls_loss = results

        # save results
        mr = ModelResults()
        if epoch is not None:
            mr.epoch = epoch
        mr.subset_id = subset_id
        mr.model_id = self.db_model.id
        mr.metric_precision = metric_precision
        mr.metric_recall = metric_recall
        mr.metric_map_50 = metric_map_50
        mr.metric_map_50_95 = metric_map_50_95
        mr.val_box_loss = val_box_loss
        mr.val_obj_loss = val_obj_loss
        mr.val_cls_loss = val_cls_loss
        db.session.add(mr)
        db.session.flush()

        # ap per class
        for index, value in enumerate(maps):
            mcr = ModelClassResult(model_results_id=mr.id, class_id=index, confidence=value)
            db.session.add(mcr)

        db.session.commit()

        return mr

    def distance(self, p1: Annotation, x, y):
        """Calculate the Euclidean distance between two points"""
        return math.sqrt((p1.x_center - x) ** 2 + (p1.y_center - y) ** 2)

    def normalize_preds(self, pred, im0, im, image_id):

        annotations = Annotation.query.filter(and_(
            Annotation.project_id == self.project_id,
            Annotation.image_id == image_id
        )).all()

        predictions = []
        for i, det in enumerate(pred):  # per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (cls, *xywh, conf)
                    class_nr = cls.item()
                    x, y, w, h = xywh
                    conf = conf.item()
                    predictions.append((x,y,w,h,conf,int(class_nr)))
        # todo update database with error detections
        if len(predictions) != len(annotations):
            return
        for x, y, w, h, conf, class_nr in predictions:
            closest_ano = min(annotations, key=lambda a: self.distance(a, x, y))
            if class_nr != closest_ano.class_id:
                break
            w_diff = abs(w - closest_ano.width)
            h_diff = abs(h - closest_ano.height)
            # print(f"{x:.3f} {y:.3f} {w:.3f} {h:.3f}")
            # print(f"{closest_ano.x_center:.3f} {closest_ano.y_center:.3f} {closest_ano.width:.3f} {closest_ano.height:.3f}")
            if w_diff > self.project_settings.pretest_size_difference_threshold:
                break
            elif h_diff > self.project_settings.pretest_size_difference_threshold:
                break
        else:
            self.good_images.add(image_id)


# needs to be here to avoid circular import error
from project.yolo.yolov5 import val, train, detect


class TrainSession:
    def __init__(self, project: Project,
                 project_settings: ProjectSettings,
                 name: str,
                 ss_test_id: int,
                 ss_train_id: int):
        self.ms_train = ModelStatus.query.filter(ModelStatus.name.like("training")).first()
        self.ms_test = ModelStatus.query.filter(ModelStatus.name.like("testing")).first()
        self.ms_ready = ModelStatus.query.filter(ModelStatus.name.like("ready")).first()
        self.ms_error = ModelStatus.query.filter(ModelStatus.name.like("error")).first()

        self.ss_test_id = ss_test_id
        self.ss_train_id = ss_train_id

        # create new model
        self.db_model = Model(model_status_id=self.ms_test.id, project_id=project.id, epochs=project_settings.epochs)

        self.prev_model = None
        self.yaml = None

        total = 0
        prev_model_id = project.latest_model_id
        if prev_model_id is not None:  # use prev model
            prev_model = Model.query.get(prev_model_id)
            model = prev_model.model
            if model is None:  # model "exists" but weights are none
                self.new_model = True
            else:
                self.prev_model = model
                total = prev_model.total_epochs
                self.new_model = False
                self.db_model.parent_model_id = prev_model_id
                #
                # with open(f"{APP_ROOT_PATH}/yolo/data/weights.pt", "wb") as binary_file:
                #     binary_file.write(self.prev_model)
        else:
            self.new_model = True

        self.db_model.total_epochs = total + project_settings.epochs

        # update database
        db.session.add(self.db_model)
        db.session.commit()
        self.project = project
        self.project_settings = project_settings

        self.good_images = set()

        # modify yolo backbone
        if self.new_model:
            with open(f"{APP_ROOT_PATH}/yolo/yolov5/models/{name}.yaml", "r") as stream:
                yaml_file = yaml.safe_load(stream)
                yaml_file["nc"] = self.project_settings.max_class_nr
            self.yaml = yaml_file
            with open(f'{APP_ROOT_PATH}/yolo/data/backbone.yaml', 'w') as outfile:
                yaml.dump(yaml_file, outfile, default_flow_style=False)

        self.stream = SqlStream(self.project.id, self.project_settings, self.ss_train_id, self.ss_test_id, self.db_model)


    def pretest(self):
        if self.new_model:
            return

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # load yolo settings
        opt = detect.parse_opt(True)

        setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/weights.pt")
        setattr(opt, "source", f"{APP_ROOT_PATH}/yolo/data/pretest_images")
        # setattr(opt, "nosave", True)
        # setattr(opt, "save_txt", True)
        # setattr(opt, "save_conf", True)
        setattr(opt, "conf_thres", self.project_settings.min_confidence_threshold)
        setattr(opt, "iou_thres", self.project_settings.min_iou_threshold)
        setattr(opt, "imgsz", [self.project_settings.img_size, self.project_settings.img_size])

        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/pretest_results")
        setattr(opt, "name", "yolo_test")

        # load model
        setattr(opt, "binary_weights", self.prev_model)

        # load sql stream
        setattr(opt, "sql_stream", self.stream)

        # run model
        detect.main(opt)

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
        # todo resize image to correct size
        images = Image.query.filter(Image.project_id == self.project.id)


        # filter out "good" images
        has_train = False
        has_test = False
        print("skipping", len(self.stream.good_images))
        for image in images.yield_per(DB_READ_BATCH_SIZE):
            if has_train and has_test:
                break

            # skip confident images
            if image.id in self.good_images:
                continue

            if image.subset_id == self.ss_train_id:
                # if has_train:
                #     continue
                # has_train = True
                if image.id in self.stream.good_images:
                    continue
                location = f"{APP_ROOT_PATH}/yolo/data/train"
            elif image.subset_id == self.ss_test_id:
                # if has_test:
                #     continue
                # has_test = True
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
        setattr(opt, "imgsz", self.project_settings.img_size)
        setattr(opt, "epochs", self.project_settings.epochs)

        # setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/model")
        setattr(opt, "name", "yolo_train")

        # TODO add option for ram storage
        # setattr(opt, "cache", "ram")
        # setattr(opt, "cache", "disk")  # dont use disk cache, super slow
        setattr(opt, "workers", NUMBER_OF_YOLO_WORKERS)

        if self.new_model:
            setattr(opt, "weights", "")
            setattr(opt, "cfg", f"{APP_ROOT_PATH}/yolo/data/backbone.yaml")
        else:
            setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/weights.pt")
            setattr(opt, "cfg", "")

        # add model id to opt
        # setattr(opt, "db_model_id", self.db_model.id)
        # setattr(opt, "db_train_subset_id", self.ss_train_id)

        try:
            train.main(opt, binary_weights=self.prev_model, yaml_dict=self.yaml, sql_stream=self.stream)  # long process
        except torch.cuda.OutOfMemoryError as e:
            print("out of gpu memory")
            print(e)
            self.db_model.model_status_id = self.ms_error.id
            db.session.add(self.db_model)
            db.session.commit()
            return True
        except cv2.error as e:
            print("out of ram")
            print(e)
            self.db_model.model_status_id = self.ms_error.id
            db.session.add(self.db_model)
            db.session.commit()
            return True

        # # read model weights
        # with open(f"{APP_ROOT_PATH}/yolo/data/model/yolo_train/weights/best.pt", "rb") as f:
        #     content = f.read()
        #     self.db_model.model = content
        self.db_model.model = self.stream.best
        return False

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
        setattr(opt, "workers", NUMBER_OF_YOLO_WORKERS)

        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "imgsz", self.project_settings.img_size)
        # setattr(opt, "source", "app/yolo/data/test")
        # setattr(opt, "nosave", True)
        # setattr(opt, "project", "app/yolo/data/results")
        # setattr(opt, "name", "yolo_test")
        # setattr(opt, "save_txt", True)
        # setattr(opt, "save_conf", True)
        if self.stream.best is None:
            setattr(opt, "binary_weights", self.prev_model)
        else:
            setattr(opt, "binary_weights", self.stream.best)


        # run model
        # todo dont save results
        settings = vars(opt)
        try:
            results, maps, t = val.run(**settings)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            self.db_model.model_status_id = self.ms_error.id
            db.session.add(self.db_model)
            db.session.commit()
            return True
        mr = self.stream.add_results_to_db(results, maps, "test")
        # t holds speeds per image, [a, b, c]
        # a - init time
        # b - inference time
        # c - nms time

        print(mr.metric_precision, mr.metric_recall, mr.metric_map_50, mr.metric_map_50_95)
        db.session.refresh(self.project)
        if self.project.times_auto_trained >= self.project_settings.maximum_auto_train_number:
            return False

        from project.services.queue_service import add_to_queue  # avoid circular import

        # see if model needs to be retrained
        needs_auto_train = mr.metric_map_50 < self.project_settings.minimal_map_50_threshold or \
                           mr.metric_map_50_95 < self.project_settings.minimal_map_50_95_threshold or \
                           mr.metric_recall < self.project_settings.minimal_recall_threshold or \
                           mr.metric_precision < self.project_settings.minimal_recall_threshold

        if needs_auto_train:
            add_to_queue(self.project.id, False)
        return False

    def cleanup(self):
        # if os.path.exists("project/yolo/data"):
        #     shutil.rmtree("project/yolo/data")

        # update model
        self.db_model.model_status_id = self.ms_ready.id
        self.project.latest_model_id = self.db_model.id
        self.project.times_auto_trained += 1
        db.session.add(self.db_model)
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


def start_training(project: Project) -> bool:
    """
    Start yolo training session with the latest data
    :returns
    True - error
    False - no error
    """
    # check settings entry
    project_settings = ProjectSettings.query.get(project.id)
    if project_settings is None:
        return True

    # check if backbone file is present
    name = InitialModel.query.get(project_settings.initial_model_id).name
    if not os.path.isfile(f"{APP_ROOT_PATH}/yolo/yolov5/models/{name}.yaml"):
        return True

    # check if project has test and train data
    ss_test = Subset.query.filter(Subset.name.like("test")).first()
    ss_train = Subset.query.filter(Subset.name.like("train")).first()

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
        return False
    if train_image is None:
        return False

    # clear dirs
    if os.path.exists(f"{APP_ROOT_PATH}/yolo/data"):
        shutil.rmtree(f"{APP_ROOT_PATH}/yolo/data")
    initialize_yolo_folders()

    print("ts 1")
    ts = TrainSession(project, project_settings, name, ss_test.id, ss_train.id)
    print("ts 3")
    ts.pretest()
    print("ts 4")
    ts.load_yaml()
    print("ts 5")
    ts.load_data()
    print("ts 6")
    error = ts.train()
    if error:
        print("Out of memory while training")
        return True
    print("ts 7")
    error = ts.test()
    if error:
        print("Out of memory while testing")
        return True
    print("ts 8")
    ts.cleanup()
    return False

