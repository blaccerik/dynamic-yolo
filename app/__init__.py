from flask import Flask
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy
load_dotenv()
import os
from app.config import Config, ProductionConfig
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


mode = os.environ["FLASK_DEBUG"]
if mode == 1:
    app.config.from_object(ProductionConfig())
    # # print(app.config.values())
    # # print(app.config.keys())
    # for i in app.config.items():
    #     print(i)

from app.views import home
