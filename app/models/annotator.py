from app import db


class Annotator(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Varchar, nullable=False, unique=True)
