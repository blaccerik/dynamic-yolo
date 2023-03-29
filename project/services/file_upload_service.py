from marshmallow import ValidationError
from sqlalchemy import and_
from werkzeug.utils import secure_filename

from project import db
from project.exceptions.user_not_authorized import UserNotAuthorized
from project.models.annotation import Annotation
from project.models.annotator import Annotator
from project.models.image import Image

from project.models.image_class import ImageClass
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.subset import Subset
from project.services.queue_service import add_to_queue


BATCH_SIZE = 100
class DictStorage:

    def __init__(self):
        self.annotations = []
        self.has_annotations = False
        self.image = None


def upload_file(file):
    """
    Upload single object to database
    :param file: db.model object
    """
    db.session.add(file)
    db.session.commit()


def _upload_images(images, location, person, ps: ProjectSettings, split):
    count = 0
    total_count = 0
    image_nr = 0
    ano_nr = 0

    train_ratio = int(len(images) * ps.train_ratio / 100)
    val_ratio = int(len(images) * (ps.train_ratio + ps.val_ratio) / 100)

    ss_test = Subset.query.filter(Subset.name.like("test")).first()
    ss_train = Subset.query.filter(Subset.name.like("train")).first()
    ss_val = Subset.query.filter(Subset.name.like("val")).first()
    if split == "test":
        subset_id = ss_test.id
    elif split == "train":
        subset_id = ss_train.id
    elif split == "val":
        subset_id = ss_val.id
    for ds in images:
        if split == "random":
            if total_count < train_ratio:  # add to train
                subset_id = ss_train.id
            elif total_count < val_ratio:  # add to val
                subset_id = ss_val.id
            else:  # add to test
                subset_id = ss_test.id
        image = ds.image
        anos = ds.annotations
        image.project_id = location
        image.subset_id = subset_id
        db.session.add(image)
        db.session.flush()
        image_nr += 1
        for a in anos:
            a.project_id = location
            a.annotator_id = person
            a.image_id = image.id
            db.session.add(a)
            ano_nr += 1

        count += 1
        if count >= BATCH_SIZE:
            db.session.commit()
            count = 0
        total_count += 1
    db.session.commit()
    return image_nr, ano_nr


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

    # users cant upload to "unknown" project
    if project_code == unknown_project.id:
        raise UserNotAuthorized("Can't upload to 'unknown' project")


    if split not in ["test", "train", "random", "val"]:
        raise ValidationError({"error": f"Unknown split {split}"})
    ps = ProjectSettings.query.get(project.id)

    max_class_nr = ps.max_class_nr
    # check if all annotations are in range
    # check that all images have unique name
    cache = {}
    for f, name in files:
        if type(f) is Image:
            if name in cache:
                entry = cache[name]
            else:
                entry = DictStorage()
            if entry.image is not None:
                raise ValidationError({'error': f'Duplicate images found: {name}'})
            entry.image = f
            cache[name] = entry

        elif type(f) is Annotation:
            if f.class_id >= max_class_nr:
                raise ValidationError({'error': f'Class id out of range: {name}'})
            if name in cache:
                entry = cache[name]
            else:
                entry = DictStorage()
            entry.has_annotations = True
            entry.annotations.append(f)
            cache[name] = entry
        elif f is None:
            if name in cache:
                entry = cache[name]
            else:
                entry = DictStorage()
            entry.has_annotations = True
            cache[name] = entry
    # filter cache for good and bad images

    passed_images = []
    failed_images = []
    for ds in cache.values():
        if ds.image is None:
            continue
        if not ds.has_annotations:
            failed_images.append(ds)
        else:
            passed_images.append(ds)

    # upload images
    passed_images_number, annotations_number = _upload_images(passed_images, project.id, annotator.id, ps, split)
    failed_images_number, _ = _upload_images(failed_images, unknown_project.id, annotator.id, ps, split)

    task_name = ""
    if (split == "train" or split == "random") and ps.always_check:
        task_name = "check"
    if split == "train" or split == "random":
        task_name = task_name + "train"
    if (split == "train" or split == "random") and ps.always_test:
        task_name = task_name + "test"

    # add to queue
    if task_name != "":
        add_to_queue(project.id, task_name)
    return passed_images_number, failed_images_number, annotations_number



def upload_class_file(file, project_code: int):
    project = Project.query.get(project_code)
    if project is None:
        raise ValidationError({"error":  f"Project not found"})
    project_settings = ProjectSettings.query.get(project_code)
    if project_settings is None:
        raise ValidationError({"error": f"Project settings not found"})
    max_class_nr = project_settings.max_class_nr
    content = file.stream.read()
    file.stream.close()
    text = str(content, "utf-8")
    if text == "":
        raise ValidationError({"error": f"Cant read this file"})
    classes = []
    for line in text.splitlines():
        try:
            nr, name = line.split(" ")
            nr = int(nr)
            if nr >= max_class_nr or nr < 0:
                raise ValidationError({"error": f"Class id is out of range"})
            classes.append((nr, name))
        except:
            raise ValidationError({"error": f"Cant read this file"})
    for nr, name in classes:
        ic = ImageClass.query.filter(and_(
            ImageClass.class_id == nr,
            ImageClass.project_id == project_code
        )).first()
        if ic is None:
            ic = ImageClass(project_id=project_code, class_id=nr, name=name)
        else:
            ic.name = name
        db.session.add(ic)
    db.session.commit()


