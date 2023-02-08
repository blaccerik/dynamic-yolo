import atexit
import os

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from flask_sqlalchemy import SQLAlchemy



db = SQLAlchemy()


# ----------------------------
# Application Factory Function
# ----------------------------

def create_app(config_filename=None):
    # Create the Flask application
    app = Flask(__name__)

    # Configure the Flask application
    #config_type = os.getenv('CONFIG_TYPE', default='config.DevelopmentConfig')
    app.config.from_object(config_filename)
    # project.debug = True
    # print(project.config)
    # for i in project.config:
    #     print(i, project.config[i])
    initialize_extensions(app)
    register_blueprints(app)

    # Check if the database needs to be initialized

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        from project.queue_manager import update_queue

        scheduler = BackgroundScheduler(job_defaults={'max_instances': 2})
        scheduler.add_job(func=update_queue, args=[app], trigger="interval", seconds=5)
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())



    return app

# ----------------
# Helper Functions
# ----------------

def initialize_extensions(app):
    db.init_app(app)
    from project.models.image import Image
    from project.models.annotation import Annotation
    from project.models.annotator import Annotator
    from project.models.image_class import ImageClass
    from project.models.queue import Queue
    from project.models.model_status import ModelStatus
    from project.models.model import Model
    from project.models.project import Project
    from project.models.model_results import ModelResults
    from project.models.project_settings import ProjectSettings
    from project.models.model_image import ModelImage
    from project.models.initial_model import InitialModel
    from project.models.image_subset import ImageSubset

    # Check if the database needs to be initialized
    recreate = False
    if recreate:
        with app.app_context():
            db.drop_all()
            db.create_all()

            # static data
            s1 = ModelStatus()
            s1.name = "training"
            s2 = ModelStatus()
            s2.name = "ready"
            s3 = ModelStatus()
            s3.name = "testing"
            db.session.add(s1)
            db.session.add(s2)
            db.session.add(s3)
            db.session.commit()

            a1 = Annotator()
            a1.name = "model"
            a2 = Annotator()
            a2.name = "human"
            db.session.add(a1)
            db.session.add(a2)
            db.session.commit()

            im1 = InitialModel(name="yolov5n")
            im2 = InitialModel(name="yolov5s")
            im3 = InitialModel(name="yolov5m")
            im4 = InitialModel(name="yolov5l")
            im5 = InitialModel(name="yolov5x")
            db.session.add_all([im1, im2, im3, im4, im5])
            db.session.commit()

            iss1 = ImageSubset(name="test")
            iss2 = ImageSubset(name="train")
            db.session.add_all([iss1, iss2])
            db.session.commit()

            p = Project(name="unknown")
            db.session.add(p)
            db.session.flush()
            ps = ProjectSettings(id=p.id, max_class_nr=80)
            db.session.add(ps)
            db.session.commit()

            # dummy data
            pro = Project(name="project")
            db.session.add(pro)
            db.session.flush()
            pros = ProjectSettings(id=pro.id, max_class_nr=80)
            db.session.add(pros)
            db.session.commit()


def register_blueprints(app):
    from project.views import home
    from project.views import upload
    from project.views import upload_classes

    app.register_blueprint(upload_classes.mod)
    app.register_blueprint(home.mod)
    app.register_blueprint(upload.mod)

# TODO Fix observer functionality