from sqlalchemy import Integer, Column, Float, ForeignKey, VARCHAR, PrimaryKeyConstraint, UniqueConstraint, \
    ForeignKeyConstraint

from app import db


class Annotation(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False)
    confidence = Column(Float, nullable=True)
    x_center = Column(Float, nullable=False)
    y_center = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    # todo figure out class ids
    # class_id = Column(Integer, ForeignKey("class.id"), nullable=False)
    class_id = Column(Integer, nullable=False)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=True)
    annotator_id = Column(Integer, ForeignKey("annotator.id"), nullable=False)
    upload_batch_id = Column(Integer, ForeignKey("upload_batch.id"), nullable=False)
