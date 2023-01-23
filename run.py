import os
from app import app
from app.api import upload_files
from app.models.image import Image
from PIL import Image as pil_image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Handler(FileSystemEventHandler):
    """Handler decides what to do in case of a change in monitored directory"""

    @staticmethod
    def on_created(event):
        files = os.listdir(event.src_path)
        with app.app_context():
            uploaded_files = [f for f in _filter_files(files, event.src_path)]
            upload_files(uploaded_files, 'test1', 'human')


def _filter_files(files, path):
    for f in files:
        for item in _convert_to_db_items(f, path):
            yield item


def _convert_to_db_items(f: str, path: str):
    if f.endswith('.jpg') or f.endswith('.png'):
        with pil_image.open(os.path.join(path, f)) as img:
            width, height = img.size
            img_binary = img.tobytes()
            name = f.split('.')[0]
            return (Image(image=img_binary, width=width, height=height, project_id=1), name),


if __name__ == '__main__':
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'images_to_upload'))
    observer.start()
    app.run(debug=False)
    observer.join()
