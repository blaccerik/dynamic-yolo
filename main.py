import argparse
import datetime
import io
import logging
import math
import os
import pathlib
import shutil
import sys
import time
from PIL import Image as Pil_Image
from matplotlib import pyplot as plt


import cv2
import numpy as np
import sqlalchemy.orm
import torch
import yaml
from dotenv import load_dotenv
from flask_sqlalchemy.session import Session
from sqlalchemy import and_, func, or_, create_engine, BigInteger, Column, VARCHAR, Integer, ForeignKey, LargeBinary, \
    Float, Boolean, DateTime, PrimaryKeyConstraint, outerjoin
from sqlalchemy.orm import aliased, sessionmaker

from yolov5.utils.general import scale_boxes, xyxy2xywh
from yolov5.utils.augmentations import letterbox

load_dotenv()
DB_READ_BATCH_SIZE = int(os.getenv("DB_READ_BATCH_SIZE"))
NUMBER_OF_YOLO_WORKERS = int(os.getenv("NUMBER_OF_YOLO_WORKERS"))
APP_ROOT_PATH = pathlib.Path(__file__).parent.resolve()
engine = create_engine(
    f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['DB_NAME']}")
Session = sessionmaker(engine)

base = sqlalchemy.orm.declarative_base()


### database models ###
class Project(base):
    __tablename__ = 'project'
    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)
    auto_train_count = Column(Integer, nullable=False, default=0)

    latest_model_id = Column(BigInteger, ForeignKey("model.id", name="fk_latest_model"), nullable=True)
    project_status_id = Column(Integer, ForeignKey("project_status.id"), nullable=False)


class Image(base):
    __tablename__ = 'image'
    id = Column(BigInteger, primary_key=True)
    image = Column(LargeBinary, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    subset_id = Column(Integer, ForeignKey("subset.id"), nullable=False)


class Annotation(base):
    __tablename__ = "annotation"
    id = Column(BigInteger, primary_key=True)
    x_center = Column(Float, nullable=False)
    y_center = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    class_id = Column(Integer, nullable=False)
    # image_class_id = Column(BigInteger, ForeignKey("image_class.id"), nullable=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("image.id"), nullable=False)
    annotator_id = Column(BigInteger, ForeignKey("annotator.id"), nullable=True)


class ProjectSettings(base):
    __tablename__ = "project_settings"
    id = Column(BigInteger, ForeignKey(Project.id), primary_key=True)
    max_class_nr = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False, default=3)
    batch_size = Column(Integer, nullable=False, default=8)
    img_size = Column(Integer, nullable=False, default=640)
    initial_model_id = Column(Integer, ForeignKey("initial_model.id"), nullable=False)

    # data splits
    train_ratio = Column(Integer, nullable=False, default=50)
    val_ratio = Column(Integer, nullable=False, default=25)

    # error detection
    check_size_difference_threshold = Column(Float, nullable=False, default=0.05)
    check_center_difference_threshold = Column(Float, nullable=False, default=0.1)
    check_error_amount_threshold = Column(Float, nullable=False, default=0.1)

    maximum_auto_train_number = Column(Integer, nullable=False, default=3)

    # min confidence for model to read image section as detection
    min_confidence_threshold = Column(Float, nullable=False, default=0.25)
    min_iou_threshold = Column(Float, nullable=False, default=0.45)

    always_test = Column(Boolean, nullable=False, default=False)
    always_check = Column(Boolean, nullable=False, default=False)

    # train speed
    freeze_backbone = Column(Boolean, nullable=False, default=False)
    use_ram = Column(Boolean, nullable=False, default=False)

    # gpu stats
    devices = Column(VARCHAR(128), nullable=False, default="0")

    # auto train
    minimal_map_50_threshold = Column(Float, nullable=False, default=0)
    minimal_map_50_95_threshold = Column(Float, nullable=False, default=0)
    minimal_precision_threshold = Column(Float, nullable=False, default=0)
    minimal_recall_threshold = Column(Float, nullable=False, default=0)


class InitialModel(base):
    __tablename__ = "initial_model"
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)


class Task(base):
    __tablename__ = "task"
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)


class Subset(base):
    __tablename__ = "subset"
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=False)


class ModelStatus(base):
    __tablename__ = "model_status"
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)


class Model(base):
    __tablename__ = "model"
    id = Column(BigInteger, primary_key=True)
    parent_model_id = Column(BigInteger, ForeignKey('model.id'), nullable=True)
    added = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    total_epochs = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False)
    model_status_id = Column(Integer, ForeignKey("model_status.id"), nullable=False)

    project_id = Column(BigInteger, ForeignKey("project.id", name="fk_project"), nullable=False)
    model = Column(LargeBinary, nullable=True)


class AnnotationError(base):
    __tablename__ = "annotation_error"
    id = Column(BigInteger, primary_key=True)
    model_annotation_id = Column(BigInteger, ForeignKey(Annotation.id), nullable=True)
    human_annotation_id = Column(BigInteger, ForeignKey(Annotation.id), nullable=True)

    confidence = Column(Float, nullable=True)

    image_count = Column(Integer, nullable=False)
    human_annotation_count = Column(Integer, nullable=True)

    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("image.id"), nullable=False)


class ModelImage(base):
    __tablename__ = "model_image"
    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("image.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint("model_id", "image_id", name="unique"),
    )


class Annotator(base):
    __tablename__ = "annotator"
    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)


class ProjectStatus(base):
    __tablename__ = "project_status"
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)


class ImageClass(base):
    __tablename__ = "image_class"
    name = Column(VARCHAR(128), nullable=False, unique=False)
    class_id = Column(Integer, nullable=False, unique=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    __table_args__ = (PrimaryKeyConstraint("class_id", "project_id"),)


class ModelResults(base):
    __tablename__ = "model_results"
    id = Column(BigInteger, primary_key=True)
    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=False)
    subset_id = Column(Integer, ForeignKey("subset.id"), nullable=False)
    epoch = Column(Integer, nullable=True)
    metric_precision = Column(Float, nullable=False, default=0)
    metric_recall = Column(Float, nullable=False, default=0)
    metric_map_50 = Column(Float, nullable=False, default=0)
    metric_map_50_95 = Column(Float, nullable=False, default=0)
    val_box_loss = Column(Float, nullable=False, default=0)
    val_obj_loss = Column(Float, nullable=False, default=0)
    val_cls_loss = Column(Float, nullable=False, default=0)


class ModelClassResult(base):
    __tablename__ = "model_class_result"
    model_results_id = Column(BigInteger, ForeignKey("model_results.id"), nullable=False)
    class_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)

    __table_args__ = (PrimaryKeyConstraint("model_results_id", "class_id"),)


### end database models ###

class Prediction:

    def __init__(self, annotation: Annotation, conf):
        self.annotation = annotation
        if conf is None:
            conf = 0
        self.conf = conf

    def __str__(self):
        return f"{self.annotation.id} {self.annotation.x_center} {self.annotation.y_center}"

class SqlStream:

    def __init__(self, project_id, project_settings, ss_val_id, ss_test_id, ss_train_id, skip_train):
        self.project_id = project_id
        self.project_settings = project_settings
        self.ss_val_id = ss_val_id
        self.ss_test_id = ss_test_id
        self.ss_train_id = ss_train_id
        self.db_model_id = None

        self.best_model_weights = None
        self.skip_train = skip_train
        self.skipped_count = 0
        self.cache = {}

        # check caching
        self.last_images = {}
        self.saved_images = []

        # anos
        self.annotation_cache = {}

        # yolo dataloader
        self.img_size = [project_settings.img_size, project_settings.img_size]
        self.stride = 32
        self.auto = True

    def save_best(self, ckpt):

        buffer = io.BytesIO()
        torch.save(ckpt, buffer)
        self.best_model_weights = buffer.getvalue()

    def get_all_train_images(self, session):

        images = session.query(Image).filter(and_(
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
            self.last_images = {}
            for image in images_batch:

                content = image.image

                # get io bytes
                img = Pil_Image.open(io.BytesIO(content))

                # Convert to BMP
                img_bmp = io.BytesIO()
                img.save(img_bmp, format='BMP')
                img_bmp.seek(0)

                # save for later
                self.last_images[image.id] = img_bmp

                nparr = np.frombuffer(img_bmp.getvalue(), np.uint8)

                # Decode numpy array into image
                im0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize

                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous
                yield im, im0, image.id

            # Commit changes to the database and start a new transaction
            session.commit()

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
        with Session() as session:
            # save results
            mr = ModelResults()
            if epoch is not None:
                mr.epoch = epoch
            mr.subset_id = subset_id
            mr.model_id = self.db_model_id
            mr.metric_precision = metric_precision
            mr.metric_recall = metric_recall
            mr.metric_map_50 = metric_map_50
            mr.metric_map_50_95 = metric_map_50_95
            mr.val_box_loss = val_box_loss
            mr.val_obj_loss = val_obj_loss
            mr.val_cls_loss = val_cls_loss
            session.add(mr)
            session.flush()

            # ap per class
            for index, value in enumerate(maps):
                mcr = ModelClassResult(model_results_id=mr.id, class_id=index, confidence=value)
                session.add(mcr)

            session.commit()

        return metric_precision, metric_recall, metric_map_50, metric_map_50_95

    def distance(self, human: Prediction, model: Prediction):
        """Calculate the Euclidean distance between two points"""
        return math.sqrt(
            (human.annotation.x_center - model.annotation.x_center) ** 2 +
            (human.annotation.y_center - model.annotation.y_center) ** 2)

    def find_closest_predictions(self, list1, list2):
        closest_distance = float('inf')
        closest_pair = None

        for pred1 in list1:
            for pred2 in list2:
                dist = self.distance(pred1, pred2)
                if dist < closest_distance:
                    closest_distance = dist
                    closest_pair = (pred1, pred2)

        return closest_pair

    def find_mappings(self, human_anos, model_anos):
        mappings = []
        while len(human_anos) > 0 and len(model_anos) > 0:
            human_ano, model_ano = self.find_closest_predictions(human_anos, model_anos)
            distance = self.distance(human_ano, model_ano)
            if distance > self.project_settings.check_center_difference_threshold:
                break
            human_anos.remove(human_ano)
            model_anos.remove(model_ano)
            mappings.append((model_ano, human_ano))

        for human_ano in human_anos:
            mappings.append((None, human_ano))

        for model_ano in model_anos:
            mappings.append((model_ano, None))
        return mappings

    def is_close_enough(self, model: Prediction, human: Prediction):
        if model is None or human is None:
            return False
        ano_human = human.annotation
        ano_model = model.annotation
        if ano_human.class_id != ano_model.class_id:
            return False
        w_diff = abs(ano_human.width - ano_model.width)
        h_diff = abs(ano_human.height - ano_model.height)
        x_diff = abs(ano_human.x_center - ano_model.x_center)
        y_diff = abs(ano_human.y_center - ano_model.y_center)
        return w_diff < self.project_settings.check_size_difference_threshold and \
            h_diff < self.project_settings.check_size_difference_threshold and \
            x_diff < self.project_settings.check_center_difference_threshold and \
            y_diff < self.project_settings.check_center_difference_threshold

    def normalize_preds(self, pred, im0, im, image_id, session):

        # all human annotations and how many times they appeared in errors
        anos = session.query(
            Annotation,
            func.count(AnnotationError.id).filter(AnnotationError.human_annotation_id == Annotation.id).label(
                "count")
        ).outerjoin(AnnotationError, Annotation.id == AnnotationError.model_annotation_id).filter(and_(
            Annotation.image_id == image_id,
            Annotation.annotator_id != None
        )).group_by(Annotation)


        # anos = session.query(
        #     Annotation,
        # ).filter(and_(
        #     Annotation.image_id == image_id,
        #     Annotation.annotator_id != None
        # ))
        annotations = [Prediction(a, x) for a, x in anos.all()]

        # how many times image has been used in training
        image_used = session.query(
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

        allowed_errors = int(len(annotations) * self.project_settings.check_error_amount_threshold)

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
                    # print(ano.image_id, ano.project_id)
                    session.add(ano)
                    predictions.append(Prediction(ano, conf))
        session.flush()
        mappings = self.find_mappings(annotations, predictions)

        # write results to database
        for m_pred, h_pred in mappings:

            # check if needs to make db entry
            if self.is_close_enough(m_pred, h_pred):
                # if pred overlapped with annotation then there is no need to add to database
                session.delete(m_pred.annotation)
                continue
            allowed_errors -= 1

            ae = AnnotationError()
            if h_pred is not None:
                ae.human_annotation_id = h_pred.annotation.id
                ae.human_annotation_count = h_pred.conf
            if m_pred is not None:
                ae.model_annotation_id = m_pred.annotation.id
                ae.confidence = m_pred .conf
            ae.model_id = self.db_model_id
            ae.image_count = image_used
            ae.image_id = image_id
            session.add(ae)
        if allowed_errors < 0 and not self.skip_train:
            self._save_image(image_id)
        elif allowed_errors >= 0:
            self.skipped_count += 1

    def _save_image(self, image_id):
        content = self.last_images[image_id]
        self.saved_images.append(image_id)
        location = f"{APP_ROOT_PATH}/data/train"
        # with open(f"{location}/images/{image_id}.png", "wb") as binary_file:
        #     binary_file.write(content)

        # write the BMP image on disk
        with open(f"{location}/images/{image_id}.bmp", 'wb') as f:
            f.write(content.read())


# needs to be here to avoid circular import error
from yolov5 import val, train, detect


class TrainSession:
    def __init__(self, project: Project,
                 project_settings: ProjectSettings,
                 name: str,
                 ss_test_id: int,
                 ss_train_id: int,
                 ss_val_id: int,
                 ms_train_id: int,
                 ms_test_id: int,
                 ms_ready_id: int,
                 ms_error_id: int,
                 task_name: str):
        self.initial_model_name = name
        self.initial_model_yaml = None
        self.is_new_model = True
        self.skip_loading_train_data = False

        self.ms_train_id = ms_train_id
        self.ms_test_id = ms_test_id
        self.ms_ready_id = ms_ready_id
        self.ms_error_id = ms_error_id

        self.ss_test_id = ss_test_id
        self.ss_train_id = ss_train_id
        self.ss_val_id = ss_val_id

        self.add_to_queue = False
        self.error = False

        self.skip_check = "check" not in task_name
        self.skip_train = "train" not in task_name
        self.skip_test = "test" not in task_name

        self.prev_model_weights = None
        self.prev_model_id = None
        self.prev_model_epochs = 0
        latest_model_id = project.latest_model_id
        if latest_model_id is not None:
            with Session() as session:
                model = session.query(Model).get(latest_model_id)
                if model.model is not None:
                    self.prev_model_weights = model.model
                    self.prev_model_id = model.id
                    self.prev_model_epochs = model.total_epochs
        # read into memory
        self.new_model_weights = None
        self.new_model_id = None

        self.project = project
        self.project_settings = project_settings

        self.stream = SqlStream(
            self.project.id,
            self.project_settings,
            self.ss_val_id,
            self.ss_test_id,
            self.ss_train_id,
            self.skip_train)

    def check(self):
        print("check")
        if self.skip_check:
            return

        # use prev model if possible
        if self.prev_model_id is not None:
            model_id = self.prev_model_id
            weights = self.prev_model_weights
        else:
            print("new", self.new_model_id)
            return

        # update model status
        with Session() as session:
            m = session.query(Model).get(model_id)
            m.model_status_id = self.ms_test_id
            session.add(m)
            session.commit()
        self.stream.db_model_id = model_id

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # load yolo settings
        opt = detect.parse_opt(True)

        # setattr(opt, "weights", f"{APP_ROOT_PATH}/data/weights.pt")
        # setattr(opt, "source", f"{APP_ROOT_PATH}/data/pretest_images")
        setattr(opt, "conf_thres", self.project_settings.min_confidence_threshold)
        setattr(opt, "iou_thres", self.project_settings.min_iou_threshold)
        setattr(opt, "imgsz", [self.project_settings.img_size, self.project_settings.img_size])

        # setattr(opt, "project", f"{APP_ROOT_PATH}/data/pretest_results")
        # setattr(opt, "name", "yolo_test")

        # load model
        setattr(opt, "binary_weights", weights)

        # load sql stream
        setattr(opt, "sql_stream", self.stream)

        # run model
        detect.main(opt)

        with Session() as session:
            m = session.query(Model).get(model_id)
            m.model_status_id = self.ms_ready_id
            session.add(m)
            session.commit()
        print("skipping", self.stream.skipped_count)

    def __init_new_db_model__(self):
        with Session() as session:

            # create new model
            new_model = Model(model_status_id=self.ms_train_id,
                                  project_id=self.project.id,
                                  epochs=self.project_settings.epochs)

            total = 0
            if self.prev_model_id is not None:  # use prev model
                total = self.prev_model_epochs
                new_model.parent_model_id = self.prev_model_id
                self.is_new_model = False

            new_model.total_epochs = total + self.project_settings.epochs
            # update database
            session.add(new_model)
            session.commit()
            self.new_model_id = new_model.id

    def load_model(self):
        self.__init_new_db_model__()
        if self.is_new_model:
            # modify yolo backbone
            with open(f"{APP_ROOT_PATH}/yolov5/models/{self.initial_model_name}.yaml", "r") as stream:
                yaml_file = yaml.safe_load(stream)
                yaml_file["nc"] = self.project_settings.max_class_nr
                self.initial_model_yaml = yaml_file
            with open(f'{APP_ROOT_PATH}/data/backbone.yaml', 'w') as outfile:
                yaml.dump(yaml_file, outfile, default_flow_style=False)

    def _load_train_data(self):
        start = time.time()
        with Session() as session:
            location = f"{APP_ROOT_PATH}/data/train"
            if self.skip_check:
                # load train data
                images = session.query(Image).filter(and_(
                    Image.project_id == self.project.id,
                    Image.subset_id == self.ss_train_id,
                ))
                image_ids = []
                for image in images.yield_per(DB_READ_BATCH_SIZE):
                    # save image
                    content = image.image
                    with open(f"{location}/images/{image.id}.png", "wb") as binary_file:
                        binary_file.write(content)
                    image_ids.append(image.id)
            else:
                image_ids = self.stream.saved_images
            for image_id in image_ids:
                annotations = session.query(Annotation).filter(and_(
                    Annotation.project_id == self.project.id,
                    Annotation.image_id == image_id,
                    Annotation.annotator_id != None
                ))
                text = ""
                # save label
                for a in annotations:
                    line = f"{a.class_id} {a.x_center} {a.y_center} {a.width} {a.height}\n"
                    text += line
                with open(f"{location}/labels/{image_id}.txt", "w") as text_file:
                    text_file.write(text)

                # create db entry
                mi = ModelImage(model_id=self.new_model_id, image_id=image_id)
                session.add(mi)
        end = time.time()
        print("train load total", end-start)

    def _load_val_data(self):
        start = time.time()
        with Session() as session:
            # load data
            images = session.query(Image).filter(and_(
                Image.project_id == self.project.id,
                Image.subset_id == self.ss_val_id
            ))
            location = f"{APP_ROOT_PATH}/data/val"
            for image in images.yield_per(DB_READ_BATCH_SIZE):
                # save image
                content = image.image
                with open(f"{location}/images/{image.id}.png", "wb") as binary_file:
                    binary_file.write(content)

                annotations = session.query(Annotation).filter(and_(
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
                mi = ModelImage(model_id=self.new_model_id, image_id=image.id)
                session.add(mi)

            session.commit()
        end = time.time()
        print("val load total", end - start)

    def load_train_yaml(self):
        # create yaml file
        data = {
            "path": f"{APP_ROOT_PATH}/data",
            "names": {},
            "train": "train",
            "val": "val"
        }
        with Session() as session:
            # if possible add names to
            for i in range(self.project_settings.max_class_nr):
                image_class = session.query(ImageClass).get((self.project.id, i))
                if image_class is None:
                    data["names"][i] = i
                else:
                    data["names"][i] = image_class.name

        with open(f'{APP_ROOT_PATH}/data/train_data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def train(self):
        print("train")
        if self.skip_train:
            return False
        # load data
        self.load_model()
        self.load_train_yaml()
        self._load_train_data()
        self._load_val_data()
        # self.load_train_data()
        self.stream.db_model_id = self.new_model_id

        # set logging to warning to see much less info at console
        # logging.getLogger("utils.general").setLevel(logging.WARNING)  # yolov5 logger
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # train model with labeled images
        opt = train.parse_opt(True)

        # change some values
        setattr(opt, "data", f"{APP_ROOT_PATH}/data/train_data.yaml")
        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "imgsz", self.project_settings.img_size)
        setattr(opt, "epochs", self.project_settings.epochs)
        setattr(opt, "device", self.project_settings.devices)

        # setattr(opt, "noval", True)  # validate only last epoch
        setattr(opt, "noplots", True)  # dont save plots
        # setattr(opt, "project", f"{APP_ROOT_PATH}/data/model")
        # setattr(opt, "name", "yolo_train")

        if self.project_settings.use_ram:
            setattr(opt, "cache", "ram")
        # setattr(opt, "cache", "disk")  # dont use disk cache, super slow
        setattr(opt, "workers", NUMBER_OF_YOLO_WORKERS)
        print("new model", self.is_new_model)
        if self.is_new_model:
            setattr(opt, "weights", "")
            setattr(opt, "cfg", f"{APP_ROOT_PATH}/data/backbone.yaml")
            w = None
        else:
            setattr(opt, "weights", f"{APP_ROOT_PATH}/data/weights.pt")
            w = self.prev_model_weights
            if self.project_settings.freeze_backbone:
                setattr(opt, "freeze", [10])
            setattr(opt, "cfg", "")

        try:
            train.main(opt, binary_weights=w, yaml_dict=self.initial_model_yaml, sql_stream=self.stream)  # long process
        except (torch.cuda.OutOfMemoryError, cv2.error, RuntimeError, AssertionError) as e:
            print(e)
            self.error = True
            with Session() as session:
                m = session.query(Model).get(self.new_model_id)
                m.model_status_id = self.ms_error_id
                session.add(m)
                session.commit()
            return

        w = self.stream.best_model_weights
        with Session() as session:
            m = session.query(Model).get(self.new_model_id)
            m.model = w
            m.model_status_id = self.ms_ready_id
            session.add(m)
            session.commit()
        self.new_model_weights = w

    def load_test_yaml(self):
        # create yaml file
        data = {
            "path": f"{APP_ROOT_PATH}/data",
            "names": {},
            "test": "test"
        }

        with Session() as session:
            # if possible add names to
            for i in range(self.project_settings.max_class_nr):
                image_class = session.query(ImageClass).get((self.project.id, i))
                if image_class is None:
                    data["names"][i] = i
                else:
                    data["names"][i] = image_class.name

        with open(f'{APP_ROOT_PATH}/data/test_data.yaml', 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)

    def load_test_data(self):
        with Session() as session:
            # load data
            images = session.query(Image).filter(and_(
                Image.project_id == self.project.id,
                Image.subset_id == self.ss_test_id,
            ))

            for image in images.yield_per(DB_READ_BATCH_SIZE):
                location = f"{APP_ROOT_PATH}/data/test"

                # save image
                content = image.image
                with open(f"{location}/images/{image.id}.png", "wb") as binary_file:
                    binary_file.write(content)

                annotations = session.query(Annotation).filter(and_(
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
        if self.error or self.skip_test:
            return
        if self.new_model_id is None:
            if self.prev_model_id is None:
                return
            else:
                weights = self.prev_model_weights
                model_id = self.prev_model_id
        else:
            weights = self.new_model_weights
            model_id = self.new_model_id

        # set new status for model
        with Session() as session:
            m = session.query(Model).get(model_id)
            m.model_status_id = self.ms_test_id
            session.add(m)
            session.commit()
        self.stream.db_model_id = model_id
        self.load_test_yaml()
        self.load_test_data()

        # set logging to warning to see much less info at console
        logging.getLogger("yolov5").setLevel(logging.ERROR)

        # load yolo settings
        opt = val.parse_opt(True)

        # setattr(opt, "weights", f"{APP_ROOT_PATH}/data/model/yolo_train/weights/best.pt")
        setattr(opt, "data", f"{APP_ROOT_PATH}/data/test_data.yaml")
        setattr(opt, "task", "test")
        # setattr(opt, "project", f"{APP_ROOT_PATH}/data/results")
        # setattr(opt, "name", "yolo_test")
        setattr(opt, "workers", NUMBER_OF_YOLO_WORKERS)

        setattr(opt, "batch_size", self.project_settings.batch_size)
        setattr(opt, "imgsz", self.project_settings.img_size)
        setattr(opt, "binary_weights", weights)
        # run model
        settings = vars(opt)
        try:
            results, maps, t = val.run(**settings)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(e)
            self.error = True
            # change model status back to ready
            # this model is still legal to use just testing failed
            with Session() as session:
                m = session.query(Model).get(model_id)
                m.model_status_id = self.ms_ready_id
                session.add(m)
                session.commit()
            return
        metric_precision, metric_recall, metric_map_50, metric_map_50_95 = self.stream.add_results_to_db(results, maps, "test")
        # t holds speeds per image, [a, b, c]
        # a - init time
        # b - inference time
        # c - nms time
        with Session() as session:
            m = session.query(Model).get(model_id)
            m.model_status_id = self.ms_ready_id
            session.add(m)
            session.commit()

        # dont auto train when it didnt train
        if self.skip_train:
            return

        # reached autotrain limit
        if self.project.auto_train_count <= 0:
            return

        # see if model needs to be retrained
        needs_auto_train = metric_map_50 < self.project_settings.minimal_map_50_threshold or \
                           metric_map_50_95 < self.project_settings.minimal_map_50_95_threshold or \
                           metric_recall < self.project_settings.minimal_recall_threshold or \
                           metric_precision < self.project_settings.minimal_recall_threshold
        if needs_auto_train:
            self.add_to_queue = True

    def cleanup(self):
        with Session() as session:
            # if project goes to error state
            # then turn off auto train for it
            if self.add_to_queue:
                count = self.project.auto_train_count
                self.project.auto_train_count = count - 1
            else:
                self.project.auto_train_count = 0
            # update model
            if self.new_model_id is not None and self.new_model_weights is not None:
                self.project.latest_model_id = self.new_model_id
            session.add(self.project)
            session.commit()


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
    create_path(f"{APP_ROOT_PATH}/data")
    create_path(f"{APP_ROOT_PATH}/data/train")
    create_path(f"{APP_ROOT_PATH}/data/train/images")
    create_path(f"{APP_ROOT_PATH}/data/train/labels")
    create_path(f"{APP_ROOT_PATH}/data/test")
    create_path(f"{APP_ROOT_PATH}/data/test/images")
    create_path(f"{APP_ROOT_PATH}/data/test/labels")
    create_path(f"{APP_ROOT_PATH}/data/val")
    create_path(f"{APP_ROOT_PATH}/data/val/images")
    create_path(f"{APP_ROOT_PATH}/data/val/labels")
    # create_path(f"{APP_ROOT_PATH}/data/pretest_images")
    # create_path(f"{APP_ROOT_PATH}/data/pretest_results")
    #
    # create_path(f"{APP_ROOT_PATH}/data/results")
    # create_path(f"{APP_ROOT_PATH}/data/model")


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def start_training(project_id: int, task_id: int) -> bool:
    """
    Start yolo training session with the latest data
    :returns
    True - error
    False - no error
    """
    with Session() as session:

        project = session.query(Project).get(project_id)
        if project is None:
            sys.exit(1)

        # check settings entry
        project_settings = session.query(ProjectSettings).get(project_id)
        if project_settings is None:
            sys.exit(1)

        # check if backbone file is present
        name = session.query(InitialModel).get(project_settings.initial_model_id).name
        if not os.path.isfile(f"{APP_ROOT_PATH}/yolov5/models/{name}.yaml"):
            sys.exit(1)

        # check task
        task = session.query(Task).get(task_id)
        if task is None:
            sys.exit(1)

        # check if project has test and train data
        ss_test = session.query(Subset).filter(Subset.name.like("test")).first()
        ss_train = session.query(Subset).filter(Subset.name.like("train")).first()
        ss_val = session.query(Subset).filter(Subset.name.like("val")).first()

        test_image = session.query(Image).filter(and_(
            Image.project_id == project_id,
            Image.subset_id == ss_test.id
        )).first()
        train_image = session.query(Image).filter(and_(
            Image.project_id == project_id,
            Image.subset_id == ss_train.id
        )).first()
        val_image = session.query(Image).filter(and_(
            Image.project_id == project_id,
            Image.subset_id == ss_val.id
        )).first()
        if "check" in task.name and train_image is None:
            sys.exit(0)
        if "train" in task.name and train_image is None:
            sys.exit(0)
        if "train" in task.name and val_image is None:
            sys.exit(0)
        if "test" in task.name and test_image is None:
            sys.exit(0)

        # clear dirs
        if os.path.exists(f"{APP_ROOT_PATH}/data"):
            shutil.rmtree(f"{APP_ROOT_PATH}/data")
        initialize_yolo_folders()

        ms_train = session.query(ModelStatus).filter(ModelStatus.name.like("training")).first()
        ms_ready = session.query(ModelStatus).filter(ModelStatus.name.like("ready")).first()
        ms_test = session.query(ModelStatus).filter(ModelStatus.name.like("testing")).first()
        ms_error = session.query(ModelStatus).filter(ModelStatus.name.like("error")).first()
    print(datetime.datetime.now())
    ts = TrainSession(
        project,
        project_settings,
        name,
        ss_test.id,
        ss_train.id,
        ss_val.id,
        ms_train.id,
        ms_test.id,
        ms_ready.id,
        ms_error.id,
        task.name)
    start = time.time()
    ts.check()
    end = time.time()
    print("check", end - start)
    start = time.time()
    ts.train()
    end = time.time()
    print("train", end - start)
    ts.test()
    ts.cleanup()
    print(datetime.datetime.now())
    if ts.error:
        sys.exit(1)
    sys.exit(0)

def super_test():
    start = time.time()
    with Session() as session:
        session.query()

        images = session.query(Image).filter(and_(
            Image.project_id == 4,
            Image.subset_id == 2
        ))
        batch_size = DB_READ_BATCH_SIZE
        start_idx = 0
        while True:
            # Fetch the next batch of images
            images_batch = images.offset(start_idx).limit(batch_size).all()
            # If there are no more images to process, break out of the loop
            if not images_batch:
                break
            image_ids = [image.id for image in images_batch]
            annotations = session.query(Annotation).filter(and_(
                Annotation.image_id.in_(image_ids),
                Annotation.annotator_id != None
            )).all()
            cache = {}
            for annotation in annotations:
                index = annotation.image_id
                if index in cache:
                    cache[index].append(annotation)
                else:
                    cache[index] = [annotation]
            for image in images_batch:
                content = image.image
                if image.id in cache:
                    anos = cache[image.id]
                else:
                    anos = []

            # Update the start index for the next batch
            start_idx += batch_size
            # break
    end = time.time()
    print("took", end - start) # 15.2 seconds
    # start = time.time()
    # with Session() as session:
    #     session.query()
    #
    #     images = session.query(Image).filter(and_(
    #         Image.project_id == 4,
    #         Image.subset_id == 2
    #     ))
    #     batch_size = DB_READ_BATCH_SIZE
    #     start_idx = 0
    #     while True:
    #         # Fetch the next batch of images
    #         images_batch = images.offset(start_idx).limit(batch_size).all()
    #         # If there are no more images to process, break out of the loop
    #         if not images_batch:
    #             break
    #         for annotation in annotations:
    #             index = annotation.image_id
    #             if index in cache:
    #                 cache[index].append(annotation)
    #             else:
    #                 cache[index] = [annotation]
    #         for image in images_batch:
    #             content = image.image
    #             anos = session.query(Annotation).filter(and_(
    #                 Annotation.image_id == image.id,
    #                 Annotation.annotator_id != None
    #             )).all()
    #
    #         # Update the start index for the next batch
    #         start_idx += batch_size
    # end = time.time()
    # print("took", end - start) # 16.5 seconds



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=int)
    parser.add_argument('--task_id', type=int)
    args = parser.parse_args()
    # super_test()
    start_training(args.project_id, args.task_id)
