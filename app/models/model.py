from sqlalchemy import Integer, Column, ForeignKey, DateTime, func, BigInteger
from sqlalchemy.orm import relationship

from app import db


class Model(db.Model):
    id = Column(BigInteger, primary_key=True)
    parent_model_id = Column(BigInteger, ForeignKey('model.id'), nullable=True)
    added = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    model_status_id = Column(Integer, ForeignKey("model_status.id"), nullable=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    latest_batch_id = Column(BigInteger, nullable=False)

    model_results_id = relationship("ModelResults", backref="model")
    parent = relationship('Model', remote_side=[id])

    # images = relationship("Image", secondary="model_image", back_populates="models")
