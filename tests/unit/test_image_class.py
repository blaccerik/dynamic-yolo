from flask_testing import TestCase

from project.models.image_class import ImageClass
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestImageClass(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.image_class = ImageClass.query.first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_name_column(self):
        expected_name = "a"

        actual_name = self.image_class.name

        assert expected_name == actual_name

    def test_class_id_column(self):
        expected_class_id = 1

        actual_class_id = self.image_class.class_id

        assert expected_class_id == actual_class_id

    def test_project_id_column(self):
        expected_project_id = 1

        actual_project_id = self.image_class.project_id

        assert expected_project_id == actual_project_id
