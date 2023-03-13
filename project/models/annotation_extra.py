from sqlalchemy import Integer, Column, Float, ForeignKey, BigInteger, PrimaryKeyConstraint

from project import db
from project.models.annotation import Annotation


class AnnotationErrors(db.Model):
    id = Column(BigInteger, primary_key=True)
    id_robot = Column(BigInteger, ForeignKey(Annotation.id), nullable=True)
    id_human = Column(BigInteger, ForeignKey(Annotation.id), nullable=True)

    confidence = Column(Float, nullable=True)
    training_amount = Column(Integer, nullable=False)