from project.models.annotation import Annotation
from project.models.annotator import Annotator
from project.models.image import Image
from project.models.image_class import ImageClass
from project.models.initial_model import InitialModel
from project.models.model import Model
from project.models.model_image import ModelImage
from project.models.model_results import ModelResults
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.queue import Queue
from project.models.project_status import ProjectStatus
import os

from project.models.subset import Subset


def create_database_for_testing(db):
    # static data
    s1 = ModelStatus()
    s1.name = "training"
    s2 = ModelStatus()
    s2.name = "ready"
    s3 = ModelStatus()
    s3.name = "testing"
    db.session.add(s1)
    db.session.add(s2)
    db.session.add(s3)
    db.session.commit()

    ps1 = ProjectStatus(name="busy")
    ps2 = ProjectStatus(name="idle")
    db.session.add_all([ps1, ps2])
    db.session.commit()

    a1 = Annotator()
    a1.name = "model"
    a2 = Annotator()
    a2.name = "human"
    db.session.add(a1)
    db.session.add(a2)
    db.session.commit()

    im1 = InitialModel(name="yolov5n")
    im2 = InitialModel(name="yolov5s")
    im3 = InitialModel(name="yolov5m")
    im4 = InitialModel(name="yolov5l")
    im5 = InitialModel(name="yolov5x")
    db.session.add_all([im1, im2, im3, im4, im5])
    db.session.commit()

    is1 = Subset()
    is1.name = "train"
    is2 = Subset()
    is2.name = "test"
    db.session.add(is1)
    db.session.add(is2)
    db.session.commit()

    p1 = Project(name="unknown")

    db.session.add(p1)
    db.session.flush()
    ps = ProjectSettings(id=p1.id, max_class_nr=80)
    db.session.add(ps)
    db.session.commit()

    queue1 = Queue(position=1, project_id=1)
    db.session.add(queue1)
    db.session.commit()
    # dummy data
    p2 = Project(name="project")
    db.session.add(p2)
    db.session.flush()
    pros = ProjectSettings(id=p2.id, max_class_nr=80)
    db.session.add(pros)
    db.session.commit()

    pro = Project(name="project2")
    db.session.add(pro)
    db.session.flush()
    pros = ProjectSettings(id=pro.id, max_class_nr=80)
    db.session.add(pros)
    db.session.commit()

    i1 = Image(image=bytes(10), height=15, width=14, project_id=p1.id, subset_id=is1.id)
    i2 = Image(image=os.urandom(200), height=2, width=2, project_id=p2.id, subset_id=is1.id)
    i3 = Image(image=os.urandom(200), height=2, width=2, project_id=p2.id, subset_id=is1.id)
    db.session.add_all([i1, i2, i3])
    db.session.flush()

    db.session.add_all([
        ImageClass(project_id=p1.id, name="a", class_id=1),
        ImageClass(project_id=p1.id, name="b", class_id=2),
        ImageClass(project_id=p2.id, name="c", class_id=2),
        ImageClass(project_id=p2.id, name="d", class_id=1)
    ])
    db.session.flush()

    a = Annotator.query.filter_by(name="human").first()
    a1 = Annotation(project_id=p1.id, image_id=i1.id, annotator_id=a.id, x_center=40, y_center=30, width=20, height=10,
                    class_id=0)
    a2 = Annotation(project_id=p1.id, image_id=i1.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0,
                    class_id=0)
    a3 = Annotation(project_id=p2.id, image_id=i2.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0,
                    class_id=0)
    db.session.add_all([a1, a2, a3])
    db.session.flush()

    ms = ModelStatus.query.filter_by(name="ready").first()
    m1 = Model(model_status_id=ms.id, project_id=p1.id, total_epochs=3, epochs=3)
    m2 = Model(model_status_id=ms.id, project_id=p1.id, total_epochs=100, epochs=100)
    db.session.add_all([m1, m2])
    db.session.flush()

    mi1 = ModelImage(model_id=m1.id, image_id=m1.id)
    mi2 = ModelImage(model_id=m1.id, image_id=m2.id)
    mi3 = ModelImage(model_id=m2.id, image_id=m1.id)
    mi4 = ModelImage(model_id=m2.id, image_id=m2.id)
    db.session.add_all([mi1, mi2, mi3, mi4])
    db.session.flush()

    mr1 = ModelResults(model_id=m1.id, subset_id=is1.id, metric_precision=0.1, metric_recall=0.2, metric_map_50=0.3,
                       metric_map_50_95=0.4, val_box_loss=0.5, val_obj_loss=0.6, val_cls_loss=0.7)
    mr2 = ModelResults(model_id=m2.id, subset_id=is1.id, metric_precision=0.2, metric_recall=0.3, metric_map_50=0.4,
                       metric_map_50_95=0.5, val_box_loss=0.6, val_obj_loss=0.7, val_cls_loss=0.8)

    db.session.add_all([mr1, mr2])
    db.session.commit()

    p1.latest_model_id = 1
