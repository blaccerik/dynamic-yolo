import logging
import os
from io import BytesIO

from app import db
from app.models.annotation import Annotation
from app.models.annotator import Annotator
from app.models.image import Image
import PIL

import app.yolo.yolov5.train as train
import yaml
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


def start_training(project_name: str):
    """
    Start yolo training session with the latest data
    """

    project = Project.query.filter_by(name=project_name).first()
    if project is None:
        return "project not found"

    # todo use only fresh images
    images = Image.query.filter_by(project_id=project.id).all()
    count = 0
    max_class_id = 0
    for image in images:
        annotations = Annotation.query.filter_by(project_id=project.id, image_id=image.id).all()

        # save image
        content = image.image
        with open(f"app/yolo/datasets/data/images/{count}.png", "wb") as binary_file:
            binary_file.write(content)
        text = ""
        for annotation in annotations:
            if annotation.class_id > max_class_id:
                max_class_id = annotation.class_id
            line = f"{annotation.class_id} {annotation.x_center} {annotation.y_center} {annotation.width} {annotation.height}\n"
            text += line
        with open(f"app/yolo/datasets/data/labels/{count}.txt", "w") as text_file:
            text_file.write(text)
        count += 1

    # create yaml file
    data = {
        "path": "../datasets/data",
        "train": "images",
        "val": "images",
        "names": {}
    }

    # todo add names for classes
    for i in range(max_class_id + 1):
        data["names"][i] = i

    with open('app/yolo/yolov5/data/data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # set logging to warning to see much less info at console
    logging.getLogger("yolov5").setLevel(logging.WARNING)

    # train model with labeled images
    opt = train.parse_opt(True)

    # change some values
    setattr(opt, "data", "app/yolo/yolov5/data/data.yaml")
    setattr(opt, "batch_size", 8)
    setattr(opt, "img", 640)
    setattr(opt, "epochs", 3)
    setattr(opt, "noval", True)  # validate only last epoch
    setattr(opt, "noplots", True)  # dont save plots
    setattr(opt, "name", "erik_test")
    setattr(opt, "weights", "")
    # opt.__setattr__("cfg", "yolov5n6.yaml")  # use untrained model
    setattr(opt, "weights", "app/yolo/yolov5/yolov5s.pt")  # use trained model
    train.main(opt)

    # delete images from disk
    for i in os.listdir("app/yolo/datasets/data/labels"):
        label = os.path.join("app/yolo/datasets/data/labels", i)
        os.remove(label)

    for i in os.listdir("app/yolo/datasets/data/images"):
        image = os.path.join("app/yolo/datasets/data/images", i)
        os.remove(image)


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

    # todo update queue manager to request project to be retrained
    return "done"
