from flask_testing import TestCase

from project.models.queue import Queue
from tests.initialize_test_database import create_database_for_testing
from project import db, create_app


class TestQueue(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.queue = Queue.query.first()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_position_column(self):
        expected_position = 1

        actual_position = self.queue.position

        assert expected_position == actual_position

    def test_project_id_column(self):
        expected_project_id = 1

        actual_project_id = self.queue.project_id

        assert expected_project_id == actual_project_id
