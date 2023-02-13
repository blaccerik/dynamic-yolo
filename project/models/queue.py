from sqlalchemy import Column, Integer, BigInteger, ForeignKey
from sqlalchemy.orm import relationship

from project import db


class Queue(db.Model):
    position = Column(Integer, primary_key=True, autoincrement=False)
    project_id = Column(BigInteger, ForeignKey('project.id'), nullable=False)
