# class Config(object):
#     SECRET_KEY = 'do-i-really-need-this'
#     FLASK_HTPASSWD_PATH = '/secret/.htpasswd'
#     FLASK_SECRET = SECRET_KEY
#     DB_HOST = 'database' # a docker link
#
# class ProductionConfig(Config):
#     FLASK_ENV = "development"
#     ENV = "development"
#     DB_HOST = 'my.production.database' # not a docker link

# import os
# basedir = os.path.abspath(os.path.dirname(__file__))
#
#
# class Config(object):
#     DEBUG = False
#     TESTING = False
#     CSRF_ENABLED = True
#     SECRET_KEY = 'this-really-needs-to-be-changed'
#     SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
#
#
# class ProductionConfig(Config):
#     DEBUG = False
#
#
# class StagingConfig(Config):
#     DEVELOPMENT = True
#     DEBUG = True
#
#
# class DevelopmentConfig(Config):
#     DEVELOPMENT = True
#     DEBUG = True
#
#
# class TestingConfig(Config):
#     TESTING = True
