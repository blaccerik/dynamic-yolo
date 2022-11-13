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


# create database
db = SQLAlchemy(app)

# all database class imports need to be after db object is created
# they all use db object as parent
# else it throws circular import error
from app.models.image import Image
from app.models.annotation import Annotation
from app.models.annotator import Annotator

# all route imports need to be imported after db and app objects are creared
# as they use them
# else it throws circular import error
from app.views import home


with app.app_context():
    inspector = inspect(db.engine)
    has_table = inspector.has_table("user")

    if not has_table:
        db.drop_all()
        db.create_all()

        # static data
        a1 = Annotator()
        a1.name = "model"
        a2 = Annotator()
        a2.name = "human"
        db.session.add(a1)
        db.session.add(a2)
        db.session.commit()

        # test data
        # i = Image()
        # i.image = os.urandom(100000)
        # db.session.add(i)
        # db.session.commit()
        #
        # a = Annotation(image=i, annotator=a2)
        # a.confidence = 0.5
        # a.x_center = 0
        # a.y_center = 0
        # a.width = 100
        # a.height = 100
        # db.session.add(a)
        # db.session.commit()
