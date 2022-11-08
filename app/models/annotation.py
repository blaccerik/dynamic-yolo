from app import db


class Annotation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    annotator = db.Column(db.Integer, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    x_center = db.Column(db.Integer, nullable=False)
    y_center = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
