from sqlalchemy import Column, Integer, VARCHAR, BigInteger
from sqlalchemy.orm import relationship

from app import db


class Annotator(db.Model):
    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)
    annotations = relationship("Annotation", backref="annotator")
