from app import db
from app.models.annotation import Annotation
from app.models.annotator import Annotator
from app.models.image import Image

from app.models.project import Project
from app.queue_manager import add_to_queue


# def link_images_and_annotations():
#     """
#     Go through Images and Annotations and find if any of them match
#     If they do create a database relation
#     """
#     ai = db.session.query(Annotation, Image) \
#         .filter(Annotation.name == Image.name,
#                 Annotation.image_id == None,
#                 Annotation.upload_batch_id == Image.upload_batch_id) \
#         .all()
#     for annotation, image in ai:
#         annotation.image_id = image.id
#     db.session.commit()



def upload_file(file):
    """
    Upload single object to database
    :param file: db.model object
    """
    db.session.add(file)
    db.session.commit()


def upload_files(files: list, project_name: str, uploader: str):
    """
    Upload multiple objects to database
    :param uploader:
    :param project_name:
    :param files: list of [db.model.Annotation | db.model.Image, file name]
    """
    annotator = Annotator.query.filter_by(name=uploader).first()
    if annotator is None:
        return "uploader not found"
    project = Project.query.filter_by(name=project_name).first()
    unknown_project = Project.query.filter_by(name="unknown").first()
    if project is None:
        return "project not found"

    batch_nr = project.latest_batch + 1

    # flush all images and mark them with unknown project id
    _dict = {}
    for f, name in files:
        if type(f) is Image:
            _dict[name] = [f, False]
            f.project_id = unknown_project.id
            f.batch_id = 0
            db.session.add(f)
    db.session.flush()

    # iterate over annotations and check if image exists with same name
    # if does make connection, also add mark to later change project id
    # else drop annotation
    for f, name in files:
        if type(f) is Annotation:
            if name in _dict:
                i, _ = _dict[name]
                _dict[name] = [i, True]
                f.image_id = i.id
                f.project_id = project.id
                f.annotator_id = annotator.id
                f.batch_id = batch_nr
                db.session.add(f)

    # if image has mark then change project id
    for f, name in files:
        if type(f) is Image:
            if _dict[name][1]:
                f.project_id = project.id
                f.batch_id = batch_nr
                db.session.add(f)

    # update model
    project.latest_batch = batch_nr
    db.session.add(project)
    db.session.commit()

    add_to_queue(project.id)
    return "done"
