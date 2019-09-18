from os import getenv

from dotenv import load_dotenv

load_dotenv()

class Config(object):
    SECRET_KEY = getenv('SECRET_KEY')
    ENV = getenv('FLASK_ENV')
