from sqlalchemy import Integer, Column, VARCHAR
from sqlalchemy.orm import relationship

from project import db


class InitialModel(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)
    project_settings = relationship("ProjectSettings", backref="initial_model")