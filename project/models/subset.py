from sqlalchemy import Integer, Column, Float, ForeignKey, BigInteger, VARCHAR
from sqlalchemy.orm import relationship

from project import db


class Subset(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=False)

    model_images = relationship("ModelImage", backref="subset")
    model_results = relationship("ModelResults", backref="subset")