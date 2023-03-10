from sqlalchemy import VARCHAR, Column, Integer, BigInteger, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from project import db
from project.models.project_status import ProjectStatus


def get_default_status_id():
    return ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first().id


class Project(db.Model):
    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)
    times_auto_trained = Column(Integer, nullable=False, default=0)

    latest_model_id = Column(BigInteger, ForeignKey("model.id", name="fk_latest_model"), nullable=True)

    project_status_id = Column(Integer, ForeignKey("project_status.id"), nullable=False, default=get_default_status_id)

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
