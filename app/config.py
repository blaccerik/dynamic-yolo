class Config(object):
    SECRET_KEY = 'do-i-really-need-this'
    FLASK_HTPASSWD_PATH = '/secret/.htpasswd'
    FLASK_SECRET = SECRET_KEY
    DB_HOST = 'database' # a docker link

class ProductionConfig(Config):
    FLASK_ENV = "development"
    ENV = "development"
    DB_HOST = 'my.production.database' # not a docker link