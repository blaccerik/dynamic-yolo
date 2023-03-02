import os

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

    image1 = Image(image=b'2', height=2, width=2, project_id=pro2.id, subset_id=iss1.id)
    image2 = Image(image=b'3', height=2, width=2, project_id=pro2.id, subset_id=iss1.id)

    db.session.add_all([image1, image2])
    db.session.flush()

    with open("tests/model_for_testing.pt", "rb") as f:
        binary_data = f.read()

    ms_ready = ModelStatus.query.filter_by(name="ready").first()
    ms_training = ModelStatus.query.filter_by(name="training").first()
    m1 = Model(model_status_id=ms_training.id, project_id=pro2.id, total_epochs=3, epochs=3, model=b'3')
    m2 = Model(model_status_id=ms_ready.id, project_id=pro2.id, total_epochs=100, epochs=100, model=binary_data)
    db.session.add_all([m1, m2])
    db.session.flush()

    a1 = Annotator()
    a1.name = "model"
    a2 = Annotator()
    a2.name = "human"
    db.session.add(a1)
    db.session.add(a2)
    db.session.commit()

    a = Annotator.query.filter_by(name="human").first()
    a1 = Annotation(project_id=pro2.id, image_id=image1.id, annotator_id=a.id, x_center=40, y_center=30, width=20,
                    height=10,
                    class_id=0)
    a2 = Annotation(project_id=pro2.id, image_id=image1.id, annotator_id=a.id, x_center=0, y_center=0, width=0,
                    height=0,
                    class_id=0)
    a3 = Annotation(project_id=pro.id, image_id=image2.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0,
                    class_id=0)
    db.session.add_all([a1, a2, a3])
    db.session.flush()

    project_for_queue1 = Project(name='queue_project1')
    project_for_queue2 = Project(name='queue_project2')
    db.session.add_all([project_for_queue1, project_for_queue2])
    db.session.flush()
    p_settings1 = ProjectSettings(id=project_for_queue1.id, max_class_nr=80)
    p_settings2 = ProjectSettings(id=project_for_queue2.id, max_class_nr=80)
    db.session.add_all([p_settings1, p_settings2])
    db.session.commit()

    queue1 = Queue(position=1, project_id=project_for_queue1.id)
    queue2 = Queue(position=2, project_id=project_for_queue2.id)
    db.session.add_all([queue1, queue2])
    db.session.commit()
