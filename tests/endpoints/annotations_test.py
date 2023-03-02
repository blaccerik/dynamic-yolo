from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from flask_testing import TestCase


class AnnotationsTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_annotations_default(self):
        response = self.client.get("/annotations/1")
        data = response.json
        expected_result = \
            {'id': 1,
             'confidence': None,
             'x_center': 0.4,
             'y_center': 0.3,
             'width': 0.2,
             'height': 0.1,
             'class_id': 0,
             'project_id': 3,
             'image_id': 1,
             'annotator_name': 'human'}

        assert response.status_code == 200
        assert data == expected_result

    def test_get_annotation_invalid_id(self):
        response = self.client.get("/annotations/999")
        data = response.json

        assert response.status_code == 404
        assert data == {'error': 'Check the annotation ID.'}

    def test_update_annotation_default(self):
        updated_fields = {'width': 0.7,
                          'x_center': 0.46}
        response = self.client.put("/annotations/1", json=updated_fields)
        updated_annotation = self.client.get("/annotations/1")

        expected_annotation = \
            {'id': 1,
             'confidence': None,
             'x_center': 0.46,
             'y_center': 0.3,
             'width': 0.7,
             'height': 0.1,
             'class_id': 0,
             'project_id': 3,
             'image_id': 1,
             'annotator_name': 'human'}

        assert response.status_code == 200
        assert updated_annotation.json == expected_annotation

    def test_update_annotation_invalid_id(self):
        updated_fields = {'width': 0.3,
                          'x_center': 0.43}
        response = self.client.put("/annotations/999", json=updated_fields)

        data = response.json

        assert response.status_code == 404
        assert data == {'error': 'Annotation not found!'}

    def test_update_annotation_wrong_field_name(self):
        updated_fields = {'size': 0.4}
        response = self.client.put("/annotations/2", json=updated_fields)
        data = response.json

        assert response.status_code == 400
        assert data == {'error': "Please check the following fields: {'size': ['Unknown field.']}"}

    def test_update_annotation_wrong_field_value(self):
        updated_fields = {'x_center': 1000}
        response = self.client.put("/annotations/2", json=updated_fields)
        data = response.json

        assert response.status_code == 400
        assert data == {'error': "Please check the following fields: {'x_center': ['Must be between 0 and 1']}"}

    def test_update_annotation_wrong_class_id(self):
        updated_fields = {'class_id': 100}
        response = self.client.put("/annotations/2", json=updated_fields)
        data = response.json

        assert response.status_code == 400
        assert data == {
            'error': "Please check the following fields: {'class_id': ['Class ID must be between 0 and 79']}"}
