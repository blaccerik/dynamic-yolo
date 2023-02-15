from project import db
from project.models.annotator import Annotator


def create_user(name: str) -> str:
    """
    Create a user
    """
    a = Annotator.query.filter(Annotator.name.like(name)).first()
    if a is not None:
        return "error"
    annotator = Annotator(name=name)
    db.session.add(annotator)
    db.session.commit()
    return "done"