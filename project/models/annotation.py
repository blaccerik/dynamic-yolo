from sqlalchemy import Integer, Column, Float, ForeignKey, VARCHAR, PrimaryKeyConstraint, UniqueConstraint, \
    ForeignKeyConstraint, BigInteger

from project import db


class Annotation(db.Model):
    id = Column(BigInteger, primary_key=True)
    confidence = Column(Float, nullable=True)
    x_center = Column(Float, nullable=False)
    y_center = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    class_id = Column(Integer, nullable=False)
    # image_class_id = Column(BigInteger, ForeignKey("image_class.id"), nullable=False)
    project_id = Column(BigInteger, ForeignKey("project.id"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("image.id"), nullable=True)
    annotator_id = Column(BigInteger, ForeignKey("annotator.id"), nullable=False)
