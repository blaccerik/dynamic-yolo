from sqlalchemy import Column, Integer, DateTime, func
from sqlalchemy.orm import relationship

from app import db


class UploadBatch(db.Model):
    __tablename__ = "upload_batch"
    id = Column(Integer, primary_key=True)
    added = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    images = relationship("Image", backref="upload_batch")
    annotations = relationship("Annotation", backref="upload_batch")
