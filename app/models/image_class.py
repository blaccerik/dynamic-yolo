from sqlalchemy import VARCHAR, Column, Integer, BigInteger, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from app import db


class ImageClass(db.Model):
    id = Column(BigInteger, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=False)
    class_id = Column(Integer, nullable=False, unique=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)

    __table_args__ = (UniqueConstraint("class_id", "project_id"),)
