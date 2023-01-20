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
from app.models.image_class import ImageClass
from app.models.model_image import ModelImage
from app.models.model_status import ModelStatus
from app.models.model import Model
from app.models.project import Project
from app.models.model_results import ModelResults

# all route imports need to be imported after db and app objects are created
# as route files use db and app object
# else it throws circular import error
from app.views import home
from app.views import upload


# def _read_names(db):
#     """
#     Read names.txt file and if needed store names into database
#     """
#     path = "classes/names.txt"
#     annotations = {}
#     with open(path, "r") as f:
#         for line in f.readlines():
#             nr, name = line.strip().split(" ", 1)
#             nr = int(nr)
#             annotations[nr] = name
#         classes = db.session.query(Class).all()
#
#         # check if every name is in database
#         try:
#             for _class in classes:
#                 del annotations[_class.id]
#         except Exception as e:  # database has more classes than is in names.txt
#             raise RuntimeError("Database mismatch")
#
#         # not all names are in database
#         if len(annotations) > 0:
#             for k, v in annotations.items():
#                 db.session.add(Class(id=k, name=v))
#             db.session.commit()


with app.app_context():
    # _read_names(db)

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

        project = Project()
        project.name = "unknown"
        db.session.add(project)
        db.session.commit()

        # dummy data
        p = Project(name="project")
        p1 = Project(name="test1")
        p2 = Project(name="test2")
        db.session.add_all([p, p1, p2])
        db.session.flush()

        i1 = Image(image=os.urandom(100), height=1, width=1, project_id=p1.id)
        i2 = Image(image=os.urandom(200), height=2, width=2, project_id=p2.id)
        i3 = Image(image=os.urandom(200), height=2, width=2, project_id=p2.id)
        db.session.add_all([i1, i2, i3])
        db.session.flush()

        db.session.add_all([
            ImageClass(project_id=p1.id, name="a", class_id=1),
            ImageClass(project_id=p1.id, name="b", class_id=2),
            ImageClass(project_id=p2.id, name="c", class_id=2),
            ImageClass(project_id=p2.id, name="d", class_id=1)
        ])
        db.session.flush()

        a = Annotator.query.filter_by(name="human").first()
        a1 = Annotation(project_id=p1.id, image_id=i1.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0, class_id=0)
        a2 = Annotation(project_id=p1.id, image_id=i1.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0, class_id=0)
        a3 = Annotation(project_id=p2.id, image_id=i2.id, annotator_id=a.id, x_center=0, y_center=0, width=0, height=0, class_id=0)
        db.session.add_all([a1, a2, a3])
        db.session.flush()

        ms = ModelStatus.query.filter_by(name="idle").first()
        m1 = Model(model_status_id=ms.id)
        m2 = Model(model_status_id=ms.id)
        db.session.add_all([m1, m2])
        db.session.flush()

        mi1 = ModelImage(model_id=m1.id, image_id=m1.id)
        mi2 = ModelImage(model_id=m1.id, image_id=m2.id)
        mi3 = ModelImage(model_id=m2.id, image_id=m1.id)
        mi4 = ModelImage(model_id=m2.id, image_id=m2.id)
        db.session.add_all([mi1, mi2, mi3, mi4])
        db.session.flush()

        mr1 = ModelResults(model_id=m1.id)
        mr2 = ModelResults(model_id=m1.id)
        mr3 = ModelResults(model_id=m2.id)
        db.session.add_all([mr1, mr2, mr3])
        db.session.commit()
