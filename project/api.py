from project import db
from project.models.annotation import Annotation
from project.models.annotator import Annotator
from project.models.image import Image

import project.yolo.yolov5.train as train
import yaml

from project.models.image_class import ImageClass
from project.models.project import Project
from project.queue_manager import add_to_queue


def upload_classes_to_db(project_name: str, classes: dict):
    """
    Upload classes to database.
    :param project_name: name of the project that the classes belong to
    :param classes: various classes {0:'dog',1:'cat'}
    """
    project = Project.query.filter_by(name=project_name).first()

    if project is None:
        return "Project is unknown"

    for class_nr, class_name in classes.items():
        i = ImageClass(project_id=project.id, name=class_name, class_id=class_nr)
        db.session.add(i)
    db.session.commit()


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

    # flush all images and mark them with unknown project id
    _dict = {}
    for f, name in files:
        if type(f) is Image:
            _dict[name] = [f, False]
            f.project_id = unknown_project.id
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
                db.session.add(f)

    # if image has mark then change project id
    for f, name in files:
        if type(f) is Image:
            if _dict[name][1]:
                f.project_id = project.id
                db.session.add(f)

    add_to_queue(project.id)
    return "done"


def check_existing_annotations(project_name: str):
    """
    Check the already existing annotations and remove them if they have a class
    that does not exist in the database
    :return:
    """
    project = Project.query.filter_by(name=project_name).first()

    annotations_to_delete = (
        db.session.query(Annotation)
        .filter(Annotation.project_id == project.id)
        .filter(
            ~Annotation.class_id.in_(db.session.query(ImageClass.class_id).filter(ImageClass.project_id == project.id)))
        .all()
    )
    for annotation in annotations_to_delete:
        db.session.delete(annotation)

    db.session.commit()
