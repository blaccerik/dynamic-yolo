import logging

from app import db
from app.models.annotation import Annotation
from app.models.annotator import Annotator
from app.models.image import Image

import app.yolo.yolov5.train as train
from app.models.project import Project

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


def start_training():
    """
    Start yolo training session with latest data
    """
    # set logging to warning to see much less info at console
    logging.getLogger("yolov5").setLevel(logging.WARNING)

    # train model with labeled images
    opt = train.parse_opt(True)

    # change some values
    opt.__setattr__("data", "app/yolo/yolov5/data/coco128.yaml")
    opt.__setattr__("batch_size", 8)
    opt.__setattr__("img", 640)
    opt.__setattr__("epochs", 3)
    opt.__setattr__("noval", True)  # validate only last epoch
    opt.__setattr__("noplots", True)  # dont save plots
    opt.__setattr__("name", "erik_test")
    opt.__setattr__("weights", "")
    # opt.__setattr__("cfg", "yolov5n6.yaml")  # use untrained model
    opt.__setattr__("weights", "app/yolo/yolov5/yolov5s.pt")  # use trained model

    train.main(opt)
    return


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
        if f.__class__ == Image:
            _dict[name] = [f, False]
            f.project_id = unknown_project.id
            db.session.add(f)
    db.session.flush()

    # iterate over annotations and check if image exists with same name
    # if does make connection, also add mark to later change project id
    # else drop annotation
    for f, name in files:
        if f.__class__ == Annotation:
            if name in _dict:
                i, _ = _dict[name]
                _dict[name] = [i, True]
                f.image_id = i.id
                f.project_id = project.id
                f.annotator_id = annotator.id
                db.session.add(f)

    # if image has mark then change project id
    for f, name in files:
        if f.__class__ == Image:
            if _dict[name][1]:
                f.project_id = project.id
                db.session.add(f)
    db.session.commit()
    return "done"
