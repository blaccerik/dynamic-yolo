import io
from flask_testing import TestCase
from project import create_app, db
from initialize_database_for_endpoints import create_database_for_testing
from PIL import Image
from io import BytesIO


class ProjectUploadTest(TestCase):

    def create_app(self):
        return create_app(config_filename='config.TestingConfig')

    def setUp(self):
        db.drop_all()
        db.create_all()
        create_database_for_testing(db)
        self.image = BytesIO()
        image = Image.new('RGB', size=(640, 640), color=(255, 0, 0))
        image.save(self.image, 'png')
        self.image.seek(0)

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    # Start of testing errors in file_upload_service.py

    def test_upload_endpoint(self):
        files = [
            (self.image, 'image1.png'),
            (io.BytesIO(b'1 0.5 0.5 0.1 0.1\n'), 'image1.txt')
        ]
        response = self.client.post('projects/2/upload',
                                    data={'uploader_name': 'human', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )
        assert response.status_code == 201
        assert response.json == {'message': 'Uploaded 1 images and 1 annotations. There were 0 failed images'}

    def test_upload_files_with_unknown_split(self):
        files = [
            (self.image, 'image1.png'),
            (io.BytesIO(b'1 0.5 0.5 0.1 0.1\n'), 'image1.txt')
        ]
        response = self.client.post('projects/2/upload',
                                    data={'uploader_name': 'human', 'split': 'unknown', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )

        assert response.json == {'error': 'Unknown split unknown'}

    def test_upload_files_with_unknown_project(self):
        files = [
            (self.image, 'image1.png'),
            (io.BytesIO(b'1 0.5 0.5 0.1 0.1\n'), 'image1.txt')
        ]
        response = self.client.post('projects/999/upload',
                                    data={'uploader_name': 'human', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )
        assert response.json == {'error': 'Project not found'}

    def test_upload_files_with_unknown_user(self):
        files = [
            (self.image, 'image1.png'),
            (io.BytesIO(b'1 0.5 0.5 0.1 0.1\n'), 'image1.txt')
        ]
        response = self.client.post('projects/5/upload',
                                    data={'uploader_name': 'alien', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )
        assert response.json == {'error': 'User not found'}

    def test_upload_into_unknown_project(self):
        files = [
            (self.image, 'image1.png'),
            (io.BytesIO(b'1 0.5 0.5 0.1 0.1\n'), 'image1.txt')
        ]
        response = self.client.post('projects/1/upload',
                                    data={'uploader_name': 'human', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )
        assert response.json == {'error': "Can't upload to 'unknown' project"}

    def test_upload_with_duplicate_images(self):
        duplicate_image = BytesIO()
        image = Image.new('RGB', size=(640, 640), color=(255, 0, 0))
        image.save(duplicate_image, 'png')
        duplicate_image.seek(0)
        files = [
            (self.image, 'image1.png'),
            (io.BytesIO(b'1 0.5 0.5 0.1 0.1\n'), 'image1.txt'),
            (duplicate_image, 'image1.png')
        ]

        response = self.client.post('projects/3/upload',
                                    data={'uploader_name': 'human', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )
        assert response.json == {'error': 'Duplicate images found: image1'}

    def test_upload_files_with_class_id_out_of_range(self):
        files = [
            (self.image, 'image1.png'),
            (io.BytesIO(b'100 0.5 0.5 0.1 0.1\n'), 'image1.txt')
        ]
        response = self.client.post('projects/3/upload',
                                    data={'uploader_name': 'human', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )
        assert response.json == {'error': 'Class id out of range: image1'}

    # Start of testing errors in views/project.py

    def test_annotation_has_right_parameters(self):
        # TODO Finish later
        files = [(io.BytesIO(b'-3 4 0.5 0.1 0.1\n'), 'image1.txt')]

        response = self.client.post('projects/3/upload',
                                    data={'uploader_name': 'human', 'split': 'train', 'files': files},
                                    content_type='multipart/form-data', buffered=True,
                                    follow_redirects=True
                                    )

        assert response.status_code == 400
