from sqlalchemy import Integer, Column, ForeignKey, DateTime, func, BigInteger, LargeBinary
from sqlalchemy.orm import relationship

from project import db


class Model(db.Model):
    id = Column(BigInteger, primary_key=True)
    parent_model_id = Column(BigInteger, ForeignKey('model.id'), nullable=True)
    added = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    total_epochs = Column(Integer,  nullable=False)
    epochs = Column(Integer,  nullable=False)
    model_status_id = Column(Integer, ForeignKey("model_status.id"), nullable=False)

    project_id = Column(BigInteger, ForeignKey("project.id", name="fk_project"), nullable=False)
    model = Column(LargeBinary, nullable=True)

    model_results = relationship("ModelResults", backref="model")
    parent = relationship('Model', remote_side=[id])

    images = relationship("Image", secondary="model_image", back_populates="models")
