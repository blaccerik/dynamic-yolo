from sqlalchemy import Integer, Column, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship

from app import db


class Model(db.Model):
    id = Column(Integer, primary_key=True)
    added = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    status_id = Column(Integer, ForeignKey("model_status.id"), nullable=False)
    images = relationship("Image", secondary="model_image", back_populates="models")
