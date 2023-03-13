from sqlalchemy import Integer, Column, BigInteger, ForeignKey, Float, PrimaryKeyConstraint

from project import db


class ModelClassResult(db.Model):
    model_results_id = Column(BigInteger, ForeignKey("model_results.id"), nullable=False)
    class_id = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)

    __table_args__ = (PrimaryKeyConstraint("model_results_id", "class_id"),)
