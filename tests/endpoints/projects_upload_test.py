import io
from flask_testing import TestCase
from unittest.mock import patch
from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from PIL import Image
from io import BytesIO
from project.models.annotation import Annotation


class ProjectUploadTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    @patch('project.services.file_upload_service.upload_files')
    def test_upload_endpoint(self, mock_upload_files):
        file_content = BytesIO()
        image = Image.new('RGB', size=(640, 640), color=(255, 0, 0))
        image.save(file_content, 'png')
        file_content.seek(0)

        files = [
            (file_content, 'image1.png'),
            (io.BytesIO(b'1 0.5 0.5 0.1 0.1\n'), 'annotation1.txt')
        ]

        response = self.client.post('projects/2/upload', data={'uploader_name': 'human', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )
        print(response.json)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json, {'message': 'Uploaded 2 images and 3 annotations. There were 1 failed images'})
        mock_upload_files.assert_called_once_with([(b'file_content', 'image1.jpg'),
                                                   (Annotation(x_center=0.5, y_center=0.5, width=0.1, height=0.1,
                                                               class_id=1, annotation_id='annotation1'),)])
