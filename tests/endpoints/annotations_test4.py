from flask_testing import TestCase

from project import db, create_app, create_database
from project.models.annotation import Annotation
from project.models.annotation_extra import AnnotationError
from project.models.annotator import Annotator
from project.models.image import Image
from project.models.model import Model
from project.models.project import Project


class AnnotationsTest4(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        create_database(self.app)
    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test1(self):
        # add some data

        p = Project(name="xddd")
        db.session.add(p)
        db.session.commit()

        i = Image(image=b'2', height=2, width=2, project_id=p.id, subset_id=1)
        db.session.add(i)
        db.session.commit()

        m = Model(project_id=p.id, total_epochs=0, epochs=0, model_status_id=1)
        db.session.add(m)
        db.session.commit()

        a = Annotator(name="erik")
        db.session.add(a)
        db.session.commit()

        a1 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id)
        a2 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id)
        a3 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id)
        a4 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id)
        a5 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id)

        b1 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id, annotator_id=a.id)
        b2 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id, annotator_id=a.id)
        b3 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id, annotator_id=a.id)
        b4 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id, annotator_id=a.id)
        b5 = Annotation(x_center=0, y_center=0, width=0, height=0, class_id=0, project_id=p.id, image_id=i.id, annotator_id=a.id)
        db.session.add_all([
            a1,a2,a3,a4,a5,b1,b2,b3,b4,b5
        ])
        db.session.commit()

        ae1 = AnnotationError(image_count=0, model_id=m.id, image_id=i.id, model_annotation_id=a1.id, human_annotation_id=b1.id)
        ae2 = AnnotationError(image_count=0, model_id=m.id, image_id=i.id, model_annotation_id=a2.id, human_annotation_id=b1.id)
        ae3 = AnnotationError(image_count=0, model_id=m.id, image_id=i.id, model_annotation_id=a3.id, human_annotation_id=b2.id)
        ae4 = AnnotationError(image_count=0, model_id=m.id, image_id=i.id, model_annotation_id=a4.id, human_annotation_id=b2.id)
        ae5 = AnnotationError(image_count=0, model_id=m.id, image_id=i.id, model_annotation_id=a5.id)
        ae6 = AnnotationError(image_count=0, model_id=m.id, image_id=i.id, human_annotation_id=b2.id)
        ae7 = AnnotationError(image_count=0, model_id=m.id, image_id=i.id, human_annotation_id=b3.id)
        db.session.add_all([
            ae1, ae2, ae3, ae4, ae5, ae6, ae7
        ])
        db.session.commit()


        json_data = {'uploader': "erik", 'keep': "both"}
        response = self.client.post(f"/annotations/{ae1.id}", json=json_data)
        assert response.status_code == 200

        ae = AnnotationError.query.all()
        assert len(ae) == 5
        assert Annotation.query.get(a1.id) is not None
        assert Annotation.query.get(a1.id).annotator_id == a.id
        assert Annotation.query.get(a2.id) is None
        assert Annotation.query.get(a3.id) is not None
        assert Annotation.query.get(a4.id) is not None
        assert Annotation.query.get(a5.id) is not None

        assert Annotation.query.get(b1.id) is not None
        assert Annotation.query.get(b2.id) is not None
        assert Annotation.query.get(b3.id) is not None
        assert Annotation.query.get(b4.id) is not None
        assert Annotation.query.get(b5.id) is not None
