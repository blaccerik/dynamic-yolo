from sqlalchemy import Integer, Column, ForeignKey, UniqueConstraint

from app import db


class ModelImage(db.Model):
    __tablename__ = "model_image"
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey("model.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    __table_args__ = (UniqueConstraint("model_id", "image_id", name="unique"),)
