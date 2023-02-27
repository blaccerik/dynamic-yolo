from project.models.image import Image
from project.models.annotation import Annotation
from project.models.annotator import Annotator
from project.models.image_class import ImageClass
from project.models.queue import Queue
from project.models.project_status import ProjectStatus
from project.models.model_status import ModelStatus
from project.models.model import Model
from project.models.project import Project
from project.models.model_results import ModelResults
from project.models.project_settings import ProjectSettings
from project.models.model_image import ModelImage
from project.models.initial_model import InitialModel
from project.models.subset import Subset


def create_database_for_testing(db):
    # static data
    ps1 = ProjectStatus(name="busy")
    ps2 = ProjectStatus(name="idle")
    ps3 = ProjectStatus(name="error")
    db.session.add_all([ps1, ps2, ps3])
    db.session.commit()

    ms1 = ModelStatus(name="ready")
    ms2 = ModelStatus(name="training")
    ms3 = ModelStatus(name="testing")
    ms4 = ModelStatus(name="error")
    db.session.add_all([ms1, ms2, ms3, ms4])
    db.session.commit()

    im1 = InitialModel(name="yolov5n")
    im2 = InitialModel(name="yolov5s")
    im3 = InitialModel(name="yolov5m")
    im4 = InitialModel(name="yolov5l")
    im5 = InitialModel(name="yolov5x")
    db.session.add_all([im1, im2, im3, im4, im5])
    db.session.commit()

    iss1 = Subset(name="test")
    iss2 = Subset(name="train")
    db.session.add_all([iss1, iss2])
    db.session.commit()

    p = Project(name="unknown")
    db.session.add(p)
    db.session.flush()
    ps = ProjectSettings(id=p.id, max_class_nr=80)
    db.session.add(ps)
    db.session.commit()

    # dummy data
    pro = Project(name="project")
    db.session.add(pro)
    db.session.flush()
    pros = ProjectSettings(id=pro.id, max_class_nr=80)
    db.session.add(pros)
    db.session.commit()

    pro2 = Project(name="project2")
    db.session.add(pro2)
    db.session.flush()
    pros2 = ProjectSettings(id=pro2.id, max_class_nr=80)
    db.session.add(pros2)
    db.session.commit()

    a1 = Annotator()
    a1.name = "model"
    a2 = Annotator()
    a2.name = "human"
    db.session.add(a1)
    db.session.add(a2)
    db.session.commit()
