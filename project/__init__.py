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
    config_type = os.getenv('CONFIG_TYPE', default='config.DevelopmentConfig')
    app.config.from_object('config.DevelopmentConfig')
    # project.debug = True
    # print(project.config)
    # for i in project.config:
    #     print(i, project.config[i])
    initialize_extensions(app)
    register_blueprints(app)

    # Check if the database needs to be initialized
    recreate = False
    if recreate:
        db.drop_all()
        db.create_all()

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        from project.queue_manager import update_queue

        scheduler = BackgroundScheduler()
        scheduler.add_job(func=update_queue, args=[app], trigger="interval", seconds=10)
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


def register_blueprints(app):
    from project.views import home
    from project.views import upload
    from project.views import upload_classes

    app.register_blueprint(upload_classes.mod)
    app.register_blueprint(home.mod)
    app.register_blueprint(upload.mod)

# TODO Fix observer functionality
# TODO Fix scheduler for queue updating