import atexit
import os
import time

from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

app = Flask(__name__)

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
from app.models.queue import Queue
from app.models.model_status import ModelStatus
from app.models.model import Model
from app.models.project import Project
from app.models.model_results import ModelResults
from app.models.project_settings import ProjectSettings
from app.models.model_image import ModelImage

# all route imports need to be imported after db and app objects are created
# as route files use db and app object
# else it throws circular import error
from app.views import home
from app.views import upload
from app.views import upload_classes

from app.queue_manager import update_queue

with app.app_context():
    # check if yolo folders are created
    path = "app/yolo/data"
    needed = ["images", "labels", "model", "results", "test"]
    for i in os.listdir(path):
        if i in needed:
            needed.remove(i)
    for i in needed:
        os.mkdir(os.path.join(path, i))

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

        p1 = Project(name="test1")
        p2 = Project(name="test2")
        db.session.add_all([p1, p2])
        db.session.flush()
        ps1 = ProjectSettings(id=p1.id, max_class_nr=80)
        ps2 = ProjectSettings(id=p2.id, max_class_nr=80)
        db.session.add_all([ps1, ps2])
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
        m1 = Model(model_status_id=ms.id, project_id=p1.id)
        m2 = Model(model_status_id=ms.id, project_id=p1.id)
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

    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=update_queue, trigger="interval", seconds=10)
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())


