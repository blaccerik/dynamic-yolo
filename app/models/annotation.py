from sqlalchemy import Integer, Column, Float, ForeignKey

from app import db


class Annotation(db.Model):
    id = Column(Integer, primary_key=True)
    confidence = Column(Float, nullable=True)
    x_center = Column(Float, nullable=False)
    y_center = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    # class_id = Column(Integer, ForeignKey("class.id"), nullable=False)
    class_nr = Column(Integer, nullable=False)
    annotator_id = Column(Integer, ForeignKey("annotator.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=True)
