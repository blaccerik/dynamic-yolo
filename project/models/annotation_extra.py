from sqlalchemy import Integer, Column, Float, ForeignKey, BigInteger, PrimaryKeyConstraint

from project import db
from project.models.annotation import Annotation


class AnnotationExtra(db.Model):
    id_robot = Column(BigInteger, ForeignKey(Annotation.id), primary_key=True)
    id_human = Column(BigInteger, ForeignKey(Annotation.id), nullable=False)

    confidence = Column(Float, nullable=False)
    training_amount = Column(Integer, nullable=False)