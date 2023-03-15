from sqlalchemy import Integer, Column, Float, ForeignKey, BigInteger
from sqlalchemy.orm import relationship

from project import db


class Annotation(db.Model):
    id = Column(BigInteger, primary_key=True)
    x_center = Column(Float, nullable=False)
    y_center = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    class_id = Column(Integer, nullable=False)
    # image_class_id = Column(BigInteger, ForeignKey("image_class.id"), nullable=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("image.id"), nullable=False)
    annotator_id = Column(BigInteger, ForeignKey("annotator.id"), nullable=True)
    model_annotation_errors = relationship('AnnotationError', foreign_keys='AnnotationError.model_annotation_id',
                                           cascade='all, delete')
    human_annotation_errors = relationship('AnnotationError', foreign_keys='AnnotationError.human_annotation_id',
                                           cascade='all, delete')

    # todo add check that both(annotator_id or model_id ) cant be null or have a value
