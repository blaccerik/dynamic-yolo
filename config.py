import os

# Determine the folder of the top-level directory of this project
BASEDIR = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    # ENV = 'development'
    # DEBUG = True
    # TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = \
        f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['DB_NAME']}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    # ENV = 'production'
    pass


class DevelopmentConfig(Config):
    print('development mode')
    ENV = 'development'
    FLASK_DEBUG = True
    DEBUG = True
    # TEMPLATES_AUTO_RELOAD = True
    #change to true later


class TestingConfig(Config):
    print('testing environment setup')
    TESTING = True
    SQLALCHEMY_DATABASE_URI = \
        SQLALCHEMY_DATABASE_URI = \
        f"postgresql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}@localhost/{os.environ['TEST_DB_NAME']}"
    WTF_CSRF_ENABLED = False
