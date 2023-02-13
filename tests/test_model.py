import os

from project.models.annotation import Annotation
from project.models.annotator import Annotator
from project.models.image import Image
from project.models.image_class import ImageClass
from project.models.model import Model
from project.models.model_image import ModelImage
from project.models.model_results import ModelResults
from project.models.model_status import ModelStatus
from project.models.project import Project
from project.models.project_settings import ProjectSettings
from project.models.queue import Queue
from project import db,create_app
from flask_testing import TestCase


class ModelTest(TestCase):


    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        s1 = ModelStatus()
        s1.name = "training"
        s2 = ModelStatus()
        s2.name = "ready"
        s3 = ModelStatus()
        s3.name = "idle"
        db.session.add(s1)
        db.session.add(s2)
        db.session.add(s3)
        db.session.commit()

        a1 = Annotator()
        a1.name = "model"
        a2 = Annotator()
        a2.name = "human"
        db.session.add(a1)
        db.session.add(a2)
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

        p1 = Project(name="test1")
        p2 = Project(name="test2")
        db.session.add_all([p1, p2])
        db.session.flush()
        ps1 = ProjectSettings(id=p1.id, max_class_nr=80)
        ps2 = ProjectSettings(id=p2.id, max_class_nr=80)
        db.session.add_all([ps1, ps2])
        db.session.flush()

        i1 = Image(image=os.urandom(100), height=1, width=1, project_id=p1.id)
        i2 = Image(image=os.urandom(200), height=2, width=2, project_id=p2.id)
        i3 = Image(image=os.urandom(200), height=2, width=2, project_id=p2.id)
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
        a1 = Annotation(project_id=p1.id, image_id=i1.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0,
                        class_id=0)
        a2 = Annotation(project_id=p1.id, image_id=i1.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0,
                        class_id=0)
        a3 = Annotation(project_id=p2.id, image_id=i2.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0,
                        class_id=0)
        db.session.add_all([a1, a2, a3])
        db.session.flush()

        ms = ModelStatus.query.filter_by(name="idle").first()
        m1 = Model(model_status_id=ms.id, project_id=p1.id)
        m2 = Model(model_status_id=ms.id, project_id=p1.id)
        db.session.add_all([m1, m2])
        db.session.flush()

        mi1 = ModelImage(model_id=m1.id, image_id=m1.id)
        mi2 = ModelImage(model_id=m1.id, image_id=m2.id)
        mi3 = ModelImage(model_id=m2.id, image_id=m1.id)
        mi4 = ModelImage(model_id=m2.id, image_id=m2.id)
        db.session.add_all([mi1, mi2, mi3, mi4])
        db.session.flush()

        mr1 = ModelResults(model_id=m1.id)
        mr2 = ModelResults(model_id=m1.id)
        mr3 = ModelResults(model_id=m2.id)
        db.session.add_all([mr1, mr2, mr3])
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()
    def test_model(self):

        model = Model.query.first()
        assert model.id == 1

