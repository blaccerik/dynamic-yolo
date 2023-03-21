import datetime
import io
import logging
import math
import os
import shutil

import cv2
import numpy as np
import torch
import yaml
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import aliased

from project import db, APP_ROOT_PATH, DB_READ_BATCH_SIZE, NUMBER_OF_YOLO_WORKERS
from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.annotator import Annotator
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
from project.models.task import Task
from project.services.user_service import create_user
from project.yolo.yolov5.utils.augmentations import letterbox
from project.yolo.yolov5.utils.general import scale_boxes, xyxy2xywh


class Prediction:

    def __init__(self, annotation, conf):
        self.annotation = annotation
        self.conf = conf


class SqlStream:

    def __init__(self, project_id, project_settings, ss_val_id, ss_test_id, ss_train_id):
        self.project_id = project_id
        self.project_settings = project_settings
        self.ss_val_id = ss_val_id
        self.ss_test_id = ss_test_id
        self.ss_train_id = ss_train_id
        self.db_model = None

        self.best_model_weights = None
        self.good_images = set()
        self.cache = {}

        # yolo dataloader
        self.img_size = [project_settings.img_size, project_settings.img_size]
        self.stride = 32
        self.auto = True

    def save_best(self, ckpt):

        buffer = io.BytesIO()
        torch.save(ckpt, buffer)
        self.best_model_weights = buffer.getvalue()

    def get_all_train_images(self):
        images = Image.query.filter(and_(
            Image.project_id == self.project_id,
            Image.subset_id == self.ss_train_id
        ))

        batch_size = DB_READ_BATCH_SIZE
        start_idx = 0
        while True:
            # Fetch the next batch of images
            images_batch = images.offset(start_idx).limit(batch_size).all()

            # If there are no more images to process, break out of the loop
            if not images_batch:
                break

            for image in images_batch:
                content = image.image
                nparr = np.frombuffer(content, np.uint8)

                # Decode numpy array into image
                im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous

                yield im, im0, image.id

            # Commit changes to the database and start a new transaction
            db.session.commit()

            # Update the start index for the next batch
            start_idx += batch_size

    def add_results_to_db(self, results, maps, subset, epoch=None):
        if subset == "val":
            subset_id = self.ss_val_id
        elif subset == "test":
            subset_id = self.ss_test_id
        else:
            raise RuntimeError("unknown subset")
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

    def distance(self, p1: Annotation, p: Prediction):
        """Calculate the Euclidean distance between two points"""
        return math.sqrt((p1.x_center - p.annotation.x_center) ** 2 + (p1.y_center - p.annotation.y_center) ** 2)

    def find_mappings(self, annotations, predictions):
        mappings = []

        if len(annotations) == 0:
            for p in predictions:
                mappings.append((p, None))
            return mappings

        # for every prediction find the closest annotation while being sorted by distance
        for prediction in sorted(predictions,
                                 key=lambda p: self.distance(min(annotations, key=lambda a: self.distance(a, p)), p)):
            closest_ano = min(annotations, key=lambda a: self.distance(a, prediction))
            # todo if distance is too large then add as seperate anos

            annotations.remove(closest_ano)
            mappings.append((prediction, closest_ano))
            if len(annotations) == 0:
                break
        for a in annotations:
            mappings.append((None, a))
        return mappings

    def is_close_enough(self, pred: Prediction, ano: Annotation):
        if pred is None:
            return False
        elif ano is None:
            return False
        ano2 = pred.annotation
        if ano2.class_id != ano.class_id:
            return False
        w_diff = abs(ano2.width - ano.width)
        h_diff = abs(ano2.height - ano.height)
        x_diff = abs(ano2.x_center - ano.x_center)
        y_diff = abs(ano2.y_center - ano.y_center)
        return w_diff < self.project_settings.check_size_difference_threshold and \
            h_diff < self.project_settings.check_size_difference_threshold and \
            x_diff < self.project_settings.check_center_difference_threshold and \
            y_diff < self.project_settings.check_center_difference_threshold

    def normalize_preds(self, pred, im0, im, image_id):

        # find all human annotations
        annotations = Annotation.query.filter(and_(
            Annotation.project_id == self.project_id,
            Annotation.image_id == image_id,
            Annotation.annotator_id != None
        )).all()

        ae = aliased(AnnotationError)
        annotations2 = db.session.query(
            Annotation.id,
            func.count(ae.human_annotation_id).label('count')
        ).join(
            ae,
            ae.human_annotation_id == Annotation.id
        ).filter(and_(
            Annotation.image_id == image_id,
            Annotation.annotator_id != None
        )).group_by(
            Annotation.id
        ).all()
        if image_id == 3:
            for i in annotations2:
                print(i)

        # how many times image has been used
        image_used = db.session.query(
            func.sum(Model.epochs)
        ).join(
            ModelImage, ModelImage.model_id == Model.id
        ).filter(
            ModelImage.image_id == image_id
        ).group_by(
            ModelImage.image_id
        ).scalar()
        if image_used is None:
            image_used = 0

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

                    ano = Annotation(
                        x_center=x,
                        y_center=y,
                        width=w,
                        height=h,
                        class_id=class_nr,

                        # metadata
                        project_id=self.project_id,
                        image_id=image_id
                    )
                    db.session.add(ano)

                    predictions.append(Prediction(ano, conf))

        db.session.flush()
        mappings = self.find_mappings(annotations, predictions)
        is_all_correct = True
        # write results to database
        for pred, ano in mappings:

            # check if needs to make db entry
            if self.is_close_enough(pred, ano):
                # if pred overlapped with annotation then there is no need to add to database
                db.session.delete(pred.annotation)
                continue
            is_all_correct = False

            # # find how many times this human ano has been in errors
            # if ano.id in self.cache:
            #     human_annotation_count = self.cache[ano.id]
            # else:
            #     count = db.session.query(func.count(AnnotationError.human_annotation_id)).filter(
            #         AnnotationError.human_annotation_id == ano.id).scalar()
            #     print(count)

            ae = AnnotationError()
            if ano is not None:
                ae.human_annotation_id = ano.id
            if pred is not None:
                ae.model_annotation_id = pred.annotation.id
                ae.confidence = pred.conf
            ae.model_id = self.db_model.id
            ae.image_count = image_used
            ae.image_id = image_id
            db.session.add(ae)

        if is_all_correct:
            self.good_images.add(image_id)


# needs to be here to avoid circular import error
from project.yolo.yolov5 import val, train, detect


class TrainSession:
    def __init__(self, project: Project,
                 project_settings: ProjectSettings,
                 name: str,
                 ss_test_id: int,
                 ss_train_id: int,
                 ss_val_id: int,
                 task_name: str):
        self.initial_model_name = name
        self.initial_model_yaml = None
        self.is_new_model = True

        self.ms_train = ModelStatus.query.filter(ModelStatus.name.like("training")).first()
        self.ms_test = ModelStatus.query.filter(ModelStatus.name.like("testing")).first()
        self.ms_ready = ModelStatus.query.filter(ModelStatus.name.like("ready")).first()
        self.ms_error = ModelStatus.query.filter(ModelStatus.name.like("error")).first()

        self.ss_test_id = ss_test_id
        self.ss_train_id = ss_train_id
        self.ss_val_id = ss_val_id

        self.skip_check = "check" not in task_name
        self.skip_train = "train" not in task_name
        self.skip_test = "test" not in task_name

        latest_model_id = project.latest_model_id
        if latest_model_id is None:
            self.db_prev_model = None
        else:
            self.db_prev_model = Model.query.get(latest_model_id)
        self.db_new_model = None
        self.project = project
        self.project_settings = project_settings

        self.stream = SqlStream(
            self.project.id,
            self.project_settings,
            self.ss_val_id,
            self.ss_test_id,
            self.ss_train_id)

    def check(self):
        print("check")
        if self.skip_check:
            return

        # use prev model if possible
        if self.db_prev_model is not None and self.db_prev_model.model is not None:
            model = self.db_prev_model
        elif self.db_new_model is not None and self.db_new_model.model is not None:
            model = self.db_new_model
        else:
            return
        model.model_status_id = self.ms_test.id
        db.session.add(model)
        db.session.commit()
        self.stream.db_model = model

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # load yolo settings
        opt = detect.parse_opt(True)

        setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/weights.pt")
        setattr(opt, "source", f"{APP_ROOT_PATH}/yolo/data/pretest_images")
        setattr(opt, "conf_thres", self.project_settings.min_confidence_threshold)
        setattr(opt, "iou_thres", self.project_settings.min_iou_threshold)
        setattr(opt, "imgsz", [self.project_settings.img_size, self.project_settings.img_size])

        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/pretest_results")
        setattr(opt, "name", "yolo_test")

        # load model
        setattr(opt, "binary_weights", model.model)

        # load sql stream
        setattr(opt, "sql_stream", self.stream)

        # run model
        detect.main(opt)

        model.model_status_id = self.ms_ready.id
        db.session.add(model)
        db.session.commit()

    def __init_new_db_model__(self, project, project_settings):
        # create new model
        self.db_new_model = Model(model_status_id=self.ms_train.id,
                                  project_id=project.id,
                                  epochs=project_settings.epochs)

        total = 0
        if self.db_prev_model is not None and self.db_prev_model.model is not None:  # use prev model
            total = self.db_prev_model.total_epochs
            self.db_new_model.parent_model_id = self.db_prev_model.id
            self.is_new_model = False

        self.db_new_model.total_epochs = total + project_settings.epochs
        # update database
        db.session.add(self.db_new_model)
        db.session.commit()

    def load_model(self):
        self.__init_new_db_model__(self.project, self.project_settings)
        if self.is_new_model:
            # modify yolo backbone
            with open(f"{APP_ROOT_PATH}/yolo/yolov5/models/{self.initial_model_name}.yaml", "r") as stream:
                yaml_file = yaml.safe_load(stream)
                yaml_file["nc"] = self.project_settings.max_class_nr
                self.initial_model_yaml = yaml_file
            with open(f'{APP_ROOT_PATH}/yolo/data/backbone.yaml', 'w') as outfile:
                yaml.dump(yaml_file, outfile, default_flow_style=False)

    def load_train_data(self):

        # load data
        images = Image.query.filter(
            and_(Image.project_id == self.project.id,
                 or_(
                     Image.subset_id == self.ss_train_id,
                     Image.subset_id == self.ss_val_id
                 )))

        for image in images.yield_per(DB_READ_BATCH_SIZE):
            if image.subset_id == self.ss_train_id:
                if image.id in self.stream.good_images:
                    continue
                location = f"{APP_ROOT_PATH}/yolo/data/train"
            elif image.subset_id == self.ss_val_id:
                location = f"{APP_ROOT_PATH}/yolo/data/val"
            else:
                raise RuntimeError("wrogn subset id")

            # save image
            content = image.image
            with open(f"{location}/images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)

            annotations = Annotation.query.filter(and_(
                Annotation.project_id == self.project.id,
                Annotation.image_id == image.id,
                Annotation.annotator_id != None
            ))
            text = ""
            # save label
            for a in annotations:
                line = f"{a.class_id} {a.x_center} {a.y_center} {a.width} {a.height}\n"
                text += line
            with open(f"{location}/labels/{image.id}.txt", "w") as text_file:
                text_file.write(text)

            # create db entry
            mi = ModelImage(model_id=self.db_new_model.id, image_id=image.id)
            db.session.add(mi)

        db.session.commit()

    def load_train_yaml(self):
        # create yaml file
        data = {
            "path": f"{APP_ROOT_PATH}/yolo/data",
            "names": {},
            "train": "train",
            "val": "val"
        }

        # if possible add names to
        for i in range(self.project_settings.max_class_nr):
            image_class = ImageClass.query.get((self.project.id, i))
            if image_class is None:
                data["names"][i] = i
            else:
                data["names"][i] = image_class.name

        with open(f'{APP_ROOT_PATH}/yolo/data/train_data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def train(self):
        print("train")
        if self.skip_train:
            return False
        # load data
        self.load_model()
        self.load_train_yaml()
        self.load_train_data()
        self.stream.db_model = self.db_new_model

        # set logging to warning to see much less info at console
        # logging.getLogger("utils.general").setLevel(logging.WARNING)  # yolov5 logger
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # train model with labeled images
        opt = train.parse_opt(True)

        # change some values
        setattr(opt, "data", f"{APP_ROOT_PATH}/yolo/data/train_data.yaml")
        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "imgsz", self.project_settings.img_size)
        setattr(opt, "epochs", self.project_settings.epochs)

        # setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/model")
        setattr(opt, "name", "yolo_train")

        if self.project_settings.use_ram:
            setattr(opt, "cache", "ram")
        # setattr(opt, "cache", "disk")  # dont use disk cache, super slow
        setattr(opt, "workers", NUMBER_OF_YOLO_WORKERS)
        print("new model", self.is_new_model)
        if self.is_new_model:
            setattr(opt, "weights", "")
            setattr(opt, "cfg", f"{APP_ROOT_PATH}/yolo/data/backbone.yaml")
            w = None
        else:
            setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/weights.pt")
            w = self.db_prev_model.model
            if self.project_settings.freeze_backbone:
                setattr(opt, "freeze", [10])
            setattr(opt, "cfg", "")

        try:
            train.main(opt, binary_weights=w, yaml_dict=self.initial_model_yaml, sql_stream=self.stream)  # long process
        except torch.cuda.OutOfMemoryError as e:
            print("out of gpu memory")
            print(e)
            self.db_new_model.model_status_id = self.ms_error.id
            db.session.add(self.db_new_model)
            db.session.commit()
            return True
        except cv2.error as e:
            print("out of ram")
            print(e)
            self.db_new_model.model_status_id = self.ms_error.id
            db.session.add(self.db_new_model)
            db.session.commit()
            return True

        self.db_new_model.model = self.stream.best_model_weights
        self.db_new_model.model_status_id = self.ms_ready.id
        db.session.add(self.db_new_model)
        db.session.commit()
        return False

    def load_test_yaml(self):
        # create yaml file
        data = {
            "path": f"{APP_ROOT_PATH}/yolo/data",
            "names": {},
            "test": "test"
        }

        # if possible add names to
        for i in range(self.project_settings.max_class_nr):
            image_class = ImageClass.query.get((self.project.id, i))
            if image_class is None:
                data["names"][i] = i
            else:
                data["names"][i] = image_class.name

        with open(f'{APP_ROOT_PATH}/yolo/data/test_data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_test_data(self):

        # load data
        images = Image.query.filter(and_(
            Image.project_id == self.project.id,
            Image.subset_id == self.ss_test_id,
        ))

        for image in images.yield_per(DB_READ_BATCH_SIZE):
            location = f"{APP_ROOT_PATH}/yolo/data/test"

            # save image
            content = image.image
            with open(f"{location}/images/{image.id}.png", "wb") as binary_file:
                binary_file.write(content)

            annotations = Annotation.query.filter(and_(
                Annotation.project_id == self.project.id,
                Annotation.image_id == image.id,
                Annotation.annotator_id != None
            ))
            text = ""
            # save label
            for a in annotations:
                line = f"{a.class_id} {a.x_center} {a.y_center} {a.width} {a.height}\n"
                text += line
            with open(f"{location}/labels/{image.id}.txt", "w") as text_file:
                text_file.write(text)

    def test(self):
        print("test")
        if self.skip_test:
            return False
        if self.db_new_model is None or self.db_new_model.model is None:
            if self.db_prev_model is None or self.db_prev_model.model is None:
                return False
            else:
                self.db_new_model = self.db_prev_model

        self.db_new_model.model_status_id = self.ms_test.id
        db.session.add(self.db_new_model)
        db.session.commit()
        self.stream.db_model = self.db_new_model

        self.load_test_yaml()
        self.load_test_data()

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # load yolo settings
        opt = val.parse_opt(True)

        setattr(opt, "weights", f"{APP_ROOT_PATH}/yolo/data/model/yolo_train/weights/best.pt")
        setattr(opt, "data", f"{APP_ROOT_PATH}/yolo/data/test_data.yaml")
        setattr(opt, "task", "test")
        setattr(opt, "project", f"{APP_ROOT_PATH}/yolo/data/results")
        setattr(opt, "name", "yolo_test")
        setattr(opt, "workers", NUMBER_OF_YOLO_WORKERS)

        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "imgsz", self.project_settings.img_size)
        setattr(opt, "binary_weights", self.db_new_model.model)
        # run model
        settings = vars(opt)
        try:
            results, maps, t = val.run(**settings)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            self.db_new_model.model_status_id = self.ms_error.id
            db.session.add(self.db_new_model)
            db.session.commit()
            return True
        mr = self.stream.add_results_to_db(results, maps, "test")
        # t holds speeds per image, [a, b, c]
        # a - init time
        # b - inference time
        # c - nms time

        self.db_new_model.model_status_id = self.ms_ready.id
        db.session.add(self.db_new_model)
        db.session.commit()

        # dont auto train when it didnt train
        if not self.skip_train:
            return False

        db.session.refresh(self.project)
        db.session.refresh(self.project_settings)
        if self.project.times_auto_trained >= self.project_settings.maximum_auto_train_number:
            return False

        from project.services.queue_service import add_to_queue  # avoid circular import

        # see if model needs to be retrained
        needs_auto_train = mr.metric_map_50 < self.project_settings.minimal_map_50_threshold or \
                           mr.metric_map_50_95 < self.project_settings.minimal_map_50_95_threshold or \
                           mr.metric_recall < self.project_settings.minimal_recall_threshold or \
                           mr.metric_precision < self.project_settings.minimal_recall_threshold

        if needs_auto_train:
            task_name = ""
            if self.project_settings.always_check:
                task_name = "check"
            task_name = task_name + "train"
            if self.project_settings.always_test:
                task_name = task_name + "test"
            add_to_queue(self.project.id, task_name, reset_counter=False)
        return False

    def cleanup(self):
        # update model
        if self.db_new_model is not None:
            self.project.latest_model_id = self.db_new_model.id
            db.session.add(self.db_new_model)
        if not self.skip_train:
            self.project.times_auto_trained += 1
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
    create_path(f"{APP_ROOT_PATH}/yolo/data/val")
    create_path(f"{APP_ROOT_PATH}/yolo/data/val/images")
    create_path(f"{APP_ROOT_PATH}/yolo/data/val/labels")

    create_path(f"{APP_ROOT_PATH}/yolo/data/pretest_images")
    create_path(f"{APP_ROOT_PATH}/yolo/data/pretest_results")

    create_path(f"{APP_ROOT_PATH}/yolo/data/results")
    create_path(f"{APP_ROOT_PATH}/yolo/data/model")


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def start_training(project: Project, task_id: int) -> bool:
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

    # check task
    task = Task.query.get(task_id)
    if task is None:
        return True

    # check if project has test and train data
    ss_test = Subset.query.filter(Subset.name.like("test")).first()
    ss_train = Subset.query.filter(Subset.name.like("train")).first()
    ss_val = Subset.query.filter(Subset.name.like("val")).first()

    test_image = Image.query.filter(and_(
        Image.project_id == project.id,
        Image.subset_id == ss_test.id
    )).first()
    train_image = Image.query.filter(and_(
        Image.project_id == project.id,
        Image.subset_id == ss_train.id
    )).first()
    val_image = Image.query.filter(and_(
        Image.project_id == project.id,
        Image.subset_id == ss_val.id
    )).first()
    if "check" in task.name and train_image is None:
        return False
    if "train" in task.name and train_image is None:
        return False
    if "train" in task.name and val_image is None:
        return False
    if "test" in task.name and test_image is None:
        return False

    # clear dirs
    if os.path.exists(f"{APP_ROOT_PATH}/yolo/data"):
        shutil.rmtree(f"{APP_ROOT_PATH}/yolo/data")
    initialize_yolo_folders()
    print(datetime.datetime.now())
    ts = TrainSession(project, project_settings, name, ss_test.id, ss_train.id, ss_val.id, task.name)

    ts.check()

    error = ts.train()
    if error:
        print("Out of memory while training")
        return True
    error = ts.test()
    if error:
        print("Out of memory while testing")
        return True
    print("ts 8")

    ts.cleanup()
    print(datetime.datetime.now())

    return False
