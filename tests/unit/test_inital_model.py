from flask_testing import TestCase

from project.models.initial_model import InitialModel
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestInitialModel(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.initial_model = InitialModel.query.filter_by(name="yolov5s").first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_name_column(self):
        expected_name = "yolov5s"

        actual_name = self.initial_model.name

        assert expected_name == actual_name

    def test_id_column(self):
        expected_id = 2

        actual_id = self.initial_model.id

        assert expected_id == actual_id

    def test_project_settings_column(self):
        expected_project_settings_size = 3

        actual_project_settings_size = len(self.initial_model.project_settings)

        assert expected_project_settings_size == actual_project_settings_size
