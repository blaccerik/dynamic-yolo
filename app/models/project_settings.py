from sqlalchemy import VARCHAR, Column, BigInteger, ForeignKey, Integer, Float
from sqlalchemy.orm import relationship

from app import db
from app.models.project import Project


class ProjectSettings(db.Model):
    id = Column(BigInteger, ForeignKey(Project.id), primary_key=True)
    max_class_nr = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False, default=3)
    batch_size = Column(Integer, nullable=False, default=8)
    img_size = Column(Integer, nullable=False, default=640)
    confidence_threshold = Column(Float, nullable=False, default=0.95)