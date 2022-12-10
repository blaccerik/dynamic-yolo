from sqlalchemy import func, LargeBinary, DateTime
from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

from app import db


class Image(db.Model):
    id = Column(Integer, primary_key=True)
    image = Column(LargeBinary, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    added_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # annotation_type = db.Column(db.Integer, nullable=False)
    annotations = relationship("Annotation", backref="image")
    models = relationship("Model", secondary="model_image", back_populates="images")
