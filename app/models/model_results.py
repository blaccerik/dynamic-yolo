from sqlalchemy import Integer, Column, BigInteger, ForeignKey

from app import db


class ModelResults(db.Model):
    id = Column(Integer, primary_key=True)
    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=False)
