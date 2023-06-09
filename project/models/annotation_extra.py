from sqlalchemy import Integer, Column, Float, ForeignKey, BigInteger, PrimaryKeyConstraint

from project import db
from project.models.annotation import Annotation


class AnnotationError(db.Model):
    id = Column(BigInteger, primary_key=True)
    model_annotation_id = Column(BigInteger, ForeignKey(Annotation.id), nullable=True)
    human_annotation_id = Column(BigInteger, ForeignKey(Annotation.id), nullable=True)

    confidence = Column(Float, nullable=True)

    image_count = Column(Integer, nullable=False)
    human_annotation_count = Column(Integer, nullable=True)

    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("image.id"), nullable=False)