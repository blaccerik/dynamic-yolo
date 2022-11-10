from sqlalchemy import Column, Integer, VARCHAR
from sqlalchemy.orm import relationship

from app import db


class Annotator(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)
    annotations = relationship("Annotation", backref="annotator")
