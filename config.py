import os

from dotenv import load_dotenv

# Determine the folder of the top-level directory of this project
BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv()

class Config(object):

    # ENV = 'development'
    # DEBUG = True
    # TESTING = False
    MAX_CONTENT_LENGTH = 1024 * 1024 * 1024 # 1gb
    JSON_SORT_KEYS = False
    SECRET_KEY = os.getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = \
        f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['DB_NAME']}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    ENV = 'production'
    # SQLALCHEMY_DATABASE_URI = \
    #     f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@{os.environ['DB_LOCATION']}/{os.environ['DB_NAME']}"
    # pass


class DevelopmentConfig(Config):
    print('development mode')
    ENV = 'development'
    FLASK_DEBUG = True
    DEBUG = True
    # TEMPLATES_AUTO_RELOAD = True
    # change to true later


class TestingConfig(Config):
    print('testing environment setup')
    TESTING = True
    SQLALCHEMY_DATABASE_URI = \
        SQLALCHEMY_DATABASE_URI = \
        f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['TEST_DB_NAME']}"
    WTF_CSRF_ENABLED = False
