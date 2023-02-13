from flask_testing import TestCase

from project.models.annotation import Annotation
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class AnnotationTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.annotation = Annotation.query.first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_primary_key(self):
        expected_id = 1

        actual_id = self.annotation.id

        assert expected_id == actual_id

    def test_annotation_project_id(self):
        expected_project_id = 1

        actual_project_id = self.annotation.project_id

        assert expected_project_id == actual_project_id

    def test_annotation_image_id(self):
        expected_image_id = 1

        actual_image_id = self.annotation.image_id

        assert expected_image_id == actual_image_id

    def test_annotator_id(self):
        expected_annotator_id = 2

        actual_annotator_id = self.annotation.annotator_id

        assert expected_annotator_id == actual_annotator_id

    def test_x_center(self):
        expected_x_center = 40

        actual_x_center = self.annotation.x_center

        assert expected_x_center == actual_x_center

    def test_y_center(self):
        expected_y_center = 30

        actual_y_center = self.annotation.y_center

        assert expected_y_center == actual_y_center

    def test_width_data(self):
        expected_width_data = 20

        actual_width_data = self.annotation.width

        assert expected_width_data == actual_width_data

    def test_height_data(self):
        expected_height_data = 10

        actual_height_data = self.annotation.height

        assert expected_height_data == actual_height_data

    def test_class_id(self):
        expected_class_id = 0

        actual_class_id = self.annotation.class_id

        assert expected_class_id == actual_class_id
