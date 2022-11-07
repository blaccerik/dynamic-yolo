import psycopg2
from flask import Flask
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import RelationshipProperty
from sqlalchemy.orm.clsregistry import _ModuleMarker

load_dotenv()
import os
# from config.py import basedir

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


app.config['SQLALCHEMY_DATABASE_URI'] = \
    f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['DB_NAME']}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


db = SQLAlchemy(app)

from app.models.User import User


with app.app_context():

    inspector = inspect(db.engine)
    has_table = inspector.has_table("user")

    if not has_table:
        db.drop_all()
        db.create_all()
        t = User()
        t.email = "eee"
        t.username = "eerrte re"

        db.session.add(t)
        db.session.commit()



from app.views import home
