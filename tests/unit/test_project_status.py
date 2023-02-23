from flask_testing import TestCase

from project.models.project_status import ProjectStatus
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestProjectStatus(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.project_status = ProjectStatus.query.filter(ProjectStatus.name.like("idle")).first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_project_status_id_column(self):
        expected_id = 2

        actual_id = self.project_status.id

        assert expected_id == actual_id

    def test_project_status_name_column(self):
        expected_project_status_name = "idle"

        actual_project_status_name = self.project_status.name

        assert expected_project_status_name == actual_project_status_name

    def test_statuses_relationship_column(self):
        expected_project_statuses_length = 3

        actual_project_statuses_length = len(self.project_status.statuses)

        assert expected_project_statuses_length == actual_project_statuses_length
