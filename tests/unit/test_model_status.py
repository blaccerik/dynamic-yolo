from flask_testing import TestCase

from project.models.model_status import ModelStatus
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestModelStatus(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.model_status = ModelStatus.query.filter_by(name="ready").first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_id_column(self):
        expected_id = 2

        actual_id = self.model_status.id

        assert expected_id == actual_id

    def test_name_column(self):
        expected_name = "ready"

        actual_name = self.model_status.name

        assert expected_name == actual_name

    def test_statuses_column(self):
        expected_statuses_model_id = 2

        actual_statuses_model_id = self.model_status.statuses[1].id

        assert expected_statuses_model_id == actual_statuses_model_id
