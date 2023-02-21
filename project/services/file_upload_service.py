from marshmallow import ValidationError

from project import db
from project.models.annotation import Annotation
from project.models.annotator import Annotator
from project.models.image import Image

from project.models.image_class import ImageClass
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.subset import Subset
from project.services.queue_service import add_to_queue


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


def _upload_images(images, location, ratio, split):
    count = 0
    threshold = int(ratio * len(images))

    ss_test = Subset.query.filter(Subset.name.like("test")).first()
    ss_train = Subset.query.filter(Subset.name.like("train")).first()
    if split == "test":
        subset_id = ss_test.id
    elif split == "train":
        subset_id = ss_train.id

    for image in images.values():
        if split == "random":
            if count > threshold:  # add to test
                subset_id = ss_test.id
            else:  # add to train
                subset_id = ss_train.id
        image.project_id = location
        image.subset_id = subset_id
        db.session.add(image)
        count += 1

    db.session.flush()  # generate ids
    return count


def upload_files(files: list, project_code: int, uploader: str, split: str) -> (int, int, int):
    """
    Upload multiple objects to database
    :param project_code:
    :param split: test, train, random
    :param uploader:
    :param files: list of [db.model.Annotation | db.model.Image, file name]
    """

    annotator = Annotator.query.filter_by(name=uploader).first()
    if annotator is None:
        raise ValidationError({"error":  f"User not found"})

    project = Project.query.get(project_code)
    unknown_project = Project.query.filter_by(name="unknown").first()
    if project is None:
        raise ValidationError({"error":  f"Project not found"})

    if split not in ["test", "train", "random"]:
        raise ValidationError({"error": f"Unknown split {split}"})
    ps = ProjectSettings.query.get(project.id)
    ratio = ps.train_test_ratio

    # find all annotations
    annotations = {x[1] for x in files if type(x[0]) is Annotation}

    # find images
    failed_images = {}
    passed_images = {}
    for f in files:
        image, name = f
        if type(image) is not Image:
            continue

        if name in annotations:
            passed_images[name] = image
        else:
            failed_images[name] = image

    # upload images
    passed_images_number = _upload_images(passed_images, project.id, ratio, split)
    failed_images_number = _upload_images(failed_images, unknown_project.id, ratio, split)
    annotations_number = 0

    # upload only passed image's annotations
    for f in files:
        ano, name = f
        if type(ano) is not Annotation:
            continue

        if name in passed_images:
            i = passed_images[name]
            ano.project_id = project.id
            ano.annotator_id = annotator.id
            ano.image_id = i.id
            db.session.add(ano)
            annotations_number += 1

    db.session.commit()
    add_to_queue(project.id)
    return passed_images_number, failed_images_number, annotations_number


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
