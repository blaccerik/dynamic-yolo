from sqlalchemy import Column, Integer, BigInteger
from sqlalchemy import LargeBinary, VARCHAR, ForeignKey
from sqlalchemy.orm import relationship

from app import db


class Image(db.Model):
    id = Column(BigInteger, primary_key=True)
    image = Column(LargeBinary, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    annotations = relationship("Annotation", backref="image")
    batch_id = Column(BigInteger, nullable=False)
    # models = relationship("Model", secondary="model_image", back_populates="images")
