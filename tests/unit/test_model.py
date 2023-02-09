from flask_testing import TestCase

from project.models.model import Model
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class ModelTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.model = Model.query.first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_epochs_column(self):
        expected_epochs = 3

        actual_epochs = self.model.epochs

        assert expected_epochs == actual_epochs

    #
    def test_total_epochs_column(self):
        expected_total_epochs = 3

        actual_total_epochs = self.model.total_epochs

        assert expected_total_epochs == actual_total_epochs

    def test_model_status_id_column(self):
        expected_model_status_id = 2

        actual_model_status_id = self.model.model_status_id

        assert expected_model_status_id == actual_model_status_id

    def test_parent_model_id_column(self):
        expected_foreign_key = None

        actual_foreign_key = self.model.parent_model_id

        assert expected_foreign_key == actual_foreign_key

    def test_id_column(self):
        expected_id = 1

        actual_id = self.model.id

        assert expected_id == actual_id
