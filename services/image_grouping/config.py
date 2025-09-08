import os
from dotenv import load_dotenv

load_dotenv()

SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")

THUMBNAIL_SAVE_PATH = os.getenv("THUMBNAIL_SAVE_PATH", "./Thumbnails")
SYSTEM_THUMBNAILS_PATH = os.getenv("SYSTEM_STORAGE")
OLD_PREFIX = os.getenv("DB_PREFIX")
NEW_PREFIX = os.getenv("CONTAINER_IMAGES_ROOT")
ROOT_PATH_thumb = os.getenv("ROOT_PATH")

os.makedirs(THUMBNAIL_SAVE_PATH, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
