import threading
import os
from app import app
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class Handler(FileSystemEventHandler):
    """Handler decides what to do in case of a change in monitored directory"""

    @staticmethod
    def on_any_event(event):
        print('event')


def run_watcher():
    """Watcher that looks for any activity inside a directory."""
    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, path=f'{os.path.dirname(os.path.abspath(__file__))}/images_to_upload')
    observer.start()
    observer.join()


if __name__ == '__main__':
    watcher_thread = threading.Thread(target=run_watcher)
    watcher_thread.start()
    app.run(debug=True)
    watcher_thread.join()
