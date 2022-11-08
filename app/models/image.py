from sqlalchemy import func

from app import db


class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary, nullable=False)
    added_at = db.Column(db.DateTime(timezone=True), server_default=func.now(), nullable=False)
    annotation_type = db.Column(db.Integer, nullable=False)
