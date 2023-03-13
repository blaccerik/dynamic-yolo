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
    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=True)
    annotation_errors_robot = relationship('AnnotationErrors', foreign_keys='AnnotationErrors.id_robot',
                                           cascade='all, delete')
    annotation_errors_human = relationship('AnnotationErrors', foreign_keys='AnnotationErrors.id_human',
                                           cascade='all, delete')

    # todo add check that both(annotator_id or model_id ) cant be null or have a value
