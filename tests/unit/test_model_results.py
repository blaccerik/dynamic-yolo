from flask_testing import TestCase

from project.models.model_results import ModelResults
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestModelResults(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.model_results = ModelResults.query.first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_id_column(self):
        expected_id = 1

        actual_id = self.model_results.id

        assert expected_id == actual_id

    def test_model_id_column(self):
        expected_model_id = 1

        actual_model_id = self.model_results.model_id

        assert expected_model_id == actual_model_id
