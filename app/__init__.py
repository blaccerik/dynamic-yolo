import os
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect

load_dotenv()

app = Flask(__name__)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql:///postgres:password@localhost/flasksql'
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# app.secret_key = 'secret string'
#
# # with app.app_context():
# db = SQLAlchemy(app)
#
# with app.app_context():
#     db.create_all()

# app.config["SQLALCHEMY_DATABASE_URI"] = 'postgresql://test:test@localhost/mydatabase'
# # app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///db.sqlite3'
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
#
# with app.app_context():
#     db = SQLAlchemy(app)
#     # db.init_app(app)
#     db.create_all()
#
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql:///postgres:password@localhost/flasksql'
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# app.secret_key = 'secret string'
#
# with app.app_context():
#     db = SQLAlchemy(app)
#     db.create_all()

# app.config['SQLALCHEMY_DATABASE_URI']='postgresql://postgres:123456@localhost:5433/flaskqna'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#
# db=SQLAlchemy(app)


# load configs
app.config['SQLALCHEMY_DATABASE_URI'] = \
    f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['DB_NAME']}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config['SECRET_KEY'] = os.environ["SECRET_KEY"]

# create database
db = SQLAlchemy(app)

# all database class imports need to be after db object is created
# they all use db object as parent
# else it throws circular import error
from app.models.image import Image
from app.models.annotation import Annotation
from app.models.annotator import Annotator
from app.models.image_class import Class
from app.models.model_image import ModelImage
from app.models.model_status import ModelStatus
from app.models.model import Model
from app.models.upload_batch import UploadBatch

# all route imports need to be imported after db and app objects are created
# as route files use db and app object
# else it throws circular import error
from app.views import home
from app.views import upload


def _read_names(db):
    """
    Read names.txt file and if needed store names into database
    """
    path = "classes/names.txt"
    annotations = {}
    with open(path, "r") as f:
        for line in f.readlines():
            nr, name = line.strip().split(" ", 1)
            nr = int(nr)
            annotations[nr] = name
        classes = db.session.query(Class).all()

        # check if every name is in database
        try:
            for _class in classes:
                del annotations[_class.id]
        except Exception as e:  # database has more classes than is in names.txt
            raise RuntimeError("Database mismatch")

        # not all names are in database
        if len(annotations) > 0:
            for k, v in annotations.items():
                db.session.add(Class(id=k, name=v))
            db.session.commit()


with app.app_context():
    _read_names(db)

    print(db.engine.table_names())
    recreate = False
    # inspector = inspect(db.engine)
    # has_table = inspector.has_table("user")
    if recreate:
        db.drop_all()
        db.create_all()

        # static data
        s1 = ModelStatus()
        s1.name = "training"
        s2 = ModelStatus()
        s2.name = "ready"
        s3 = ModelStatus()
        s3.name = "idle"
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
