from sqlalchemy import VARCHAR, Column, Integer, BigInteger, ForeignKey, UniqueConstraint, PrimaryKeyConstraint
from sqlalchemy.orm import relationship

from project import db


class ImageClass(db.Model):
    name = Column(VARCHAR(128), nullable=False, unique=False)
    class_id = Column(Integer, nullable=False, unique=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    __table_args__ = (PrimaryKeyConstraint("class_id", "project_id"),)
