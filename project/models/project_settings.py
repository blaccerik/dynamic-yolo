from select import select

from sqlalchemy import VARCHAR, Column, BigInteger, ForeignKey, Integer, Float, text
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
    confidence_threshold = Column(Float, nullable=False, default=0.95) # how confident model needs to be to skip image
    train_test_ratio = Column(Float, nullable=False, default=0.75)
    minimal_map_50_threshold = Column(Float, nullable=False, default=0)

    # min confidence for model to read image section as detection
    min_confidence_threshold = Column(Float, nullable=False, default=0.25)
    min_iou_threshold = Column(Float, nullable=False, default=0.45)



