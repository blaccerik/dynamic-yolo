from sqlalchemy import func, LargeBinary, DateTime, VARCHAR, ForeignKey, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

from app import db


class Image(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False)
    image = Column(LargeBinary, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    upload_batch_id = Column(Integer, ForeignKey("upload_batch.id"), nullable=False)
    annotations = relationship("Annotation", backref="image")
    models = relationship("Model", secondary="model_image", back_populates="images")
