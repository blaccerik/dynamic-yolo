from sqlalchemy import Integer, Column, VARCHAR
from sqlalchemy.orm import relationship

from project import db


class Task(db.Model):
    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(128), nullable=False, unique=True)
    queues = relationship("Queue", backref="task")