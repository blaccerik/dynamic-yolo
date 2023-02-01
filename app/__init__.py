import os

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

    initialize_extensions(app)
    register_blueprints(app)

    # Check if the database needs to be initialized
    recreate = False
    if recreate:
        db.drop_all()
        db.create_all()
    return app


# ----------------
# Helper Functions
# ----------------

def initialize_extensions(app):
    db.init_app(app)
    from app.models.image import Image
    from app.models.annotation import Annotation
    from app.models.annotator import Annotator
    from app.models.image_class import ImageClass
    from app.models.queue import Queue
    from app.models.model_status import ModelStatus
    from app.models.model import Model
    from app.models.project import Project
    from app.models.model_results import ModelResults
    from app.models.project_settings import ProjectSettings
    from app.models.model_image import ModelImage


def register_blueprints(app):
    from app.views import home
    from app.views import upload
    from app.views import upload_classes

    app.register_blueprint(upload_classes.mod)
    app.register_blueprint(home.mod)
    app.register_blueprint(upload.mod)

# TODO Fix observer functionality
# TODO Fix scheduler for queue updating
