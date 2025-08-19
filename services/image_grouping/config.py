import os
from dotenv import load_dotenv

load_dotenv()

SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")

THUMBNAIL_SAVE_PATH = os.getenv("THUMBNAIL_SAVE_PATH", "./Thumbnails")

os.makedirs(THUMBNAIL_SAVE_PATH, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
