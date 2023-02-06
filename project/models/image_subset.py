from sqlalchemy import Integer, Column, Float, ForeignKey, BigInteger, VARCHAR
from sqlalchemy.orm import relationship

from project import db


class ImageSubset(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=False)

    model_images = relationship("ModelImage", backref="imagesubset")