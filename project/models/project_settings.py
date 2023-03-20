from select import select

from sqlalchemy import VARCHAR, Column, BigInteger, ForeignKey, Integer, Float, text, Boolean
from sqlalchemy.orm import relationship

from project import db
from project.models.project import Project
from project.models.initial_model import InitialModel


def get_default_name_id():
    return InitialModel.query.filter(InitialModel.name.like("yolov5s")).first().id


class ProjectSettings(db.Model):
    id = Column(BigInteger, ForeignKey(Project.id), primary_key=True)
    max_class_nr = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False, default=3)
    batch_size = Column(Integer, nullable=False, default=8)
    img_size = Column(Integer, nullable=False, default=640)
    initial_model_id = Column(Integer, ForeignKey("initial_model.id"), nullable=False, default=get_default_name_id)

    # data splits
    train_ratio = Column(Integer, nullable=False, default=50)
    val_ratio = Column(Integer, nullable=False, default=25)

    # error detection
    pretest_size_difference_threshold = Column(Float, nullable=False, default=0.05)
    maximum_auto_train_number = Column(Integer, nullable=False, default=3)

    # min confidence for model to read image section as detection
    min_confidence_threshold = Column(Float, nullable=False, default=0.25)
    min_iou_threshold = Column(Float, nullable=False, default=0.45)

    always_test = Column(Boolean, nullable=False, default=False)
    always_check = Column(Boolean, nullable=False, default=False)

    # train speed
    freeze_backbone = Column(Boolean, nullable=False, default=False)
    use_ram = Column(Boolean, nullable=False, default=False)

    # auto train
    minimal_map_50_threshold = Column(Float, nullable=False, default=0)
    minimal_map_50_95_threshold = Column(Float, nullable=False, default=0)
    minimal_precision_threshold = Column(Float, nullable=False, default=0)
    minimal_recall_threshold = Column(Float, nullable=False, default=0)




