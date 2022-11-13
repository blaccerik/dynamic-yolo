from sqlalchemy import Integer, Column, Float, ForeignKey

from app import db


class Annotation(db.Model):
    id = Column(Integer, primary_key=True)
    confidence = Column(Float, nullable=False)
    x_center = Column(Integer, nullable=False)
    y_center = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    annotator_id = Column(Integer, ForeignKey("annotator.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
