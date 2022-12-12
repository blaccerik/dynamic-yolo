from app import db
from app.models.annotation import Annotation
from app.models.annotator import Annotator


def _add_item(item):
    db.session.add(item)


def upload_file(file):
    _add_item(file)
    db.session.commit()


def upload_files(files, add_human=True):
    if add_human:
        annotator = Annotator.query.filter_by(name='human').first()
    for f in files:
        if add_human and f.__class__ == Annotation:
            f.annotator_id = annotator.id
        _add_item(f)
    db.session.commit()
