from sqlalchemy import VARCHAR, Column, Integer, Float
from sqlalchemy.orm import relationship

from app import db


class Class(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)
    # classes = relationship("Annotation", backref="class")
