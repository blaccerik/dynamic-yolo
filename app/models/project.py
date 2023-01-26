from sqlalchemy import VARCHAR, Column, Integer, BigInteger, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from app import db


class Project(db.Model):
    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)

    latest_model_id = Column(BigInteger, ForeignKey("model.id", name="fk_latest_model"), nullable=True)

    queue = relationship('Queue', backref='project', lazy=True, uselist=False)
    project_settings = relationship("ProjectSettings", backref='project', lazy=True, uselist=False)
    image_classes = relationship("ImageClass", backref="project")
    images = relationship("Image", backref="project")
    annotations = relationship("Annotation", backref="project")
    models = relationship("Model", backref="project", primaryjoin="Project.id == Model.project_id")

    latest_model = relationship("Model",
                                foreign_keys=[latest_model_id],
                                uselist=False,
                                primaryjoin="Model.id==Project.latest_model_id",
                                post_update=True)
