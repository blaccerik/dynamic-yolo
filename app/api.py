from app import db
from app.models.annotation import Annotation
from app.models.annotator import Annotator
from app.models.image import Image

from app.models.upload_batch import UploadBatch


def _add_item(item):
    db.session.add(item)


def link_images_and_annotations():
    """
    Go through Images and Annotations and find if any of them match
    If they do create a relation
    """
    ai = db.session.query(Annotation, Image) \
        .filter(Annotation.name == Image.name,
                Annotation.image_id == None,
                Annotation.upload_batch_id == Image.upload_batch_id) \
        .all()
    for annotation, image in ai:
        annotation.image_id = image.id
    db.session.commit()


def upload_file(file):
    """
    Upload single object to database
    :param file: db.model object
    """
    _add_item(file)
    db.session.commit()


def upload_files(files, add_human=True):
    """
    Upload multiple objects to database
    :param files: db.model objects
    :param add_human: if true then add human annotator to only Annotation object (creates relation)
    """

    ub = UploadBatch()
    db.session.add(ub)
    db.session.flush()

    if add_human:
        annotator = Annotator.query.filter_by(name='human').first()
    for f in files:
        if add_human and f.__class__ == Annotation:
            f.annotator_id = annotator.id
        f.upload_batch_id = ub.id
        _add_item(f)
    db.session.commit()
