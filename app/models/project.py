from sqlalchemy import VARCHAR, Column, Integer, BigInteger, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from app import db


class Project(db.Model):
    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)

    latest_batch = Column(BigInteger, nullable=False)

    # todo add relation with model if possible
    latest_model_id = Column(BigInteger, nullable=True)

    queue = relationship('Queue', backref='project', lazy=True, uselist=False)
    image_classes = relationship("ImageClass", backref="project")
    images = relationship("Image", backref="project")
    annotations = relationship("Annotation", backref="project")
    models = relationship("Model", backref="project")
