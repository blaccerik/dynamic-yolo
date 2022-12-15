from sqlalchemy import Integer, Column, ForeignKey, UniqueConstraint, VARCHAR, PrimaryKeyConstraint

from app import db


class ModelImage(db.Model):
    __tablename__ = "model_image"
    model_id = Column(Integer, ForeignKey("model.id"), nullable=False)
    image_id = Column(Integer, ForeignKey("image.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint("model_id", "image_id", name="unique"),
    )
