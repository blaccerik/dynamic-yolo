from sqlalchemy import Integer, Column, ForeignKey, UniqueConstraint, VARCHAR, PrimaryKeyConstraint, BigInteger

from project import db


class ModelImage(db.Model):
    __tablename__ = "model_image"
    model_id = Column(BigInteger, ForeignKey("model.id"), nullable=False)
    image_id = Column(BigInteger, ForeignKey("image.id"), nullable=False)
    __table_args__ = (
        PrimaryKeyConstraint("model_id", "image_id", name="unique"),
    )
