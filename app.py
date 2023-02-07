# import csv
# import os
# import shutil
#
# from PIL import Image as pil_image
# from watchdog.events import FileSystemEventHandler
# from watchdog.observers import Observer
# from flask import Flask
# from project import project
# from project.api import upload_files
# from project.models.annotation import Annotation
# from project.models.image import Image
# from project.models.project import Project
#
# # File types that we are interested in processing
# image_types = (".jpeg", ".jpg", ".png")
# text_types = (".txt")
#
# from flask_sqlalchemy import SQLAlchemy
# db = SQLAlchemy()
# class Handler(FileSystemEventHandler):
#     """
#     Handler decides what to do in case of a change in monitored directory
#     """
#
#     @staticmethod
#     def on_created(event):
#         with project.app_context():
#             project_name = event.src_path.split('/')[-1]
#             uploader = "human"
#             does_project_exist = Project.query.filter_by(name=project_name).first()
#
#             if does_project_exist:
#                 files = os.listdir(event.src_path)
#                 uploaded_files = [f for f in _filter_files(files, event.src_path)]
#                 upload_files(uploaded_files, project_name, uploader)
#                 shutil.rmtree(event.src_path)
#             else:
#                 shutil.rmtree(event.src_path)
#
#
# def _filter_files(files, path):
#     for f in files:
#         if f.endswith(image_types) or f.endswith(text_types):
#             for item in _convert_to_db_items(f, path):
#                 yield item
#
#
# def _text_to_annotations(path, name):
#     _list = []
#     with open(path, 'r') as file:
#         reader = csv.reader(file, delimiter=' ')
#         for row in reader:
#             nr = int(row[0])
#             x = float(row[1])
#             y = float(row[2])
#             w = float(row[3])
#             h = float(row[4])
#             _list.append((Annotation(x_center=x, y_center=y, width=w, height=h, class_id=nr), name))
#     return _list
#
#
# def _convert_to_db_items(f: str, path: str):
#     full_path = os.path.join(path, f)
#     file_name = f.split('.')[0]
#
#     if f.endswith(image_types):
#         with pil_image.open(full_path) as img:
#             width, height = img.size
#             img_binary = img.tobytes()
#             return (Image(image=img_binary, width=width, height=height), file_name),
#     else:
#         return _text_to_annotations(full_path, file_name)
#
# def create_app():
#     project = Flask(__name__)
#     project.config['TESTING'] = True
#     project.config["SQLALCHEMY_DATABASE_URI"] = "xxxxxxtestdatabasexxx"
#     # Dynamically bind SQLAlchemy to application
#     db.init_app(project)
#     project.app_context().push()  # this does the binding
#     return project
# def create_test_app():
#     pass
# if __name__ == '__main__':
#     event_handler = Handler()
#     observer = Observer()
#     observer.schedule(event_handler, path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images_to_upload'))
#     observer.start()
#     project.run(debug=False)
#     observer.join()
#
#
import atexit

from project import create_app
app = create_app('config.DevelopmentConfig')
