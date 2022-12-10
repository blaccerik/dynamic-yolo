from sqlalchemy import Integer, Column, ForeignKey
from sqlalchemy.orm import relationship

from app import db


class Model(db.Model):
    id = Column(Integer, primary_key=True)
    version = Column(Integer, nullable=False, unique=True)
    status_id = Column(Integer, ForeignKey("model_status.id"), nullable=False)
    images = relationship("Image", secondary="model_image", back_populates="models")
