import atexit
import os
import pathlib

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.serving import is_running_from_reloader
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

db = SQLAlchemy()

# /home/...../dynamic-yolo/project
APP_ROOT_PATH = pathlib.Path(__file__).parent.resolve()

### swagger specific ###
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Dynamic yolo swagger"
    }
)

### end swagger specific ###

def create_app(config_filename=None):
    # Create the Flask application
    app = Flask(__name__)

    if config_filename is not None:
        app.config.from_object(config_filename)
    else:
        app.config.from_object('config.ProductionConfig')

    initialize_extensions(app)
    register_blueprints(app)
    initialize_extensions(app)
    register_cli_commands(app)
    init_swagger(app)

    if not is_running_from_reloader():
        from project.services.queue_service import update_queue

        scheduler = BackgroundScheduler(job_defaults={'max_instances': 2})
        scheduler.add_job(func=update_queue, args=[app], trigger="interval", seconds=5)
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())

    return app


def initialize_extensions(app):
    db.init_app(app)
    from project.models.image import Image
    from project.models.annotation import Annotation
    from project.models.annotator import Annotator
    from project.models.image_class import ImageClass
    from project.models.queue import Queue
    from project.models.project_status import ProjectStatus
    from project.models.model_status import ModelStatus
    from project.models.model import Model
    from project.models.project import Project
    from project.models.model_results import ModelResults
    from project.models.project_settings import ProjectSettings
    from project.models.model_image import ModelImage
    from project.models.initial_model import InitialModel
    from project.models.subset import Subset


def register_blueprints(app):
    from project.views import home
    from project.views import upload
    from project.views import upload_classes
    from project.views import users
    from project.views import project
    from project.views import queue
    from project.views import image
    from project.views import model
    from project.views import annotation
    from project.views import model_result

    from project.exceptions import project_not_found
    from project.exceptions import user_not_authorized
    from project.exceptions import validation_error

    app.register_blueprint(upload_classes.mod)
    app.register_blueprint(home.mod)
    app.register_blueprint(upload.REQUEST_API)
    app.register_blueprint(users.REQUEST_API)
    app.register_blueprint(project.REQUEST_API)
    app.register_blueprint(queue.REQUEST_API)
    app.register_blueprint(image.REQUEST_API)
    app.register_blueprint(model.REQUEST_API)
    app.register_blueprint(model_result.REQUEST_API)

    app.register_blueprint(annotation.REQUEST_API)

    app.register_blueprint(project_not_found.project_not_found_error)
    app.register_blueprint(user_not_authorized.user_not_authorized_error)
    app.register_blueprint(validation_error.validation_error)


def init_swagger(app):
    @app.route("/static/<path:path>")
    def send_static(path):
        """
        Swagger path
        """
        return send_from_directory("static", path)

def register_cli_commands(app):
    @app.cli.command('init_db')
    def initialize_database():
        """Initialize the database."""
        create_database(app)
        print('Database created!')

    app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)


def create_database(app):
    from project.models.image import Image
    from project.models.annotation import Annotation
    from project.models.annotator import Annotator
    from project.models.image_class import ImageClass
    from project.models.queue import Queue
    from project.models.project_status import ProjectStatus
    from project.models.model_status import ModelStatus
    from project.models.model import Model
    from project.models.project import Project
    from project.models.model_results import ModelResults
    from project.models.project_settings import ProjectSettings
    from project.models.model_image import ModelImage
    from project.models.initial_model import InitialModel
    from project.models.subset import Subset

    with app.app_context():
        db.drop_all()
        db.create_all()

        # static data
        ps1 = ProjectStatus(name="busy")
        ps2 = ProjectStatus(name="idle")
        ps3 = ProjectStatus(name="error")
        db.session.add_all([ps1, ps2, ps3])
        db.session.commit()

        ms1 = ModelStatus(name="ready")
        ms2 = ModelStatus(name="training")
        ms3 = ModelStatus(name="testing")
        ms4 = ModelStatus(name="error")
        db.session.add_all([ms1, ms2, ms3, ms4])
        db.session.commit()

        im1 = InitialModel(name="yolov5n")
        im2 = InitialModel(name="yolov5s")
        im3 = InitialModel(name="yolov5m")
        im4 = InitialModel(name="yolov5l")
        im5 = InitialModel(name="yolov5x")
        db.session.add_all([im1, im2, im3, im4, im5])
        db.session.commit()

        iss1 = Subset(name="test")
        iss2 = Subset(name="train")
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

        pro2 = Project(name="project2")
        db.session.add(pro2)
        db.session.flush()
        pros2 = ProjectSettings(id=pro2.id, max_class_nr=80)
        db.session.add(pros2)
        db.session.commit()

        a1 = Annotator()
        a1.name = "model"
        a2 = Annotator()
        a2.name = "human"
        db.session.add(a1)
        db.session.add(a2)
        db.session.commit()

# TODO Fix observer functionality
