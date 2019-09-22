# adapted from https://github.com/miguelgrinberg/flask-video-streaming
from pathlib import Path
import threading
import time
import os

import cv2 as cv
import numpy as np
import skimage.io

if os.getenv('CAMERA') == 'pi':
    import io
    import picamera

from .sudoku import Sudoku
from .sudoku_cv import detect_board, reduce_image_size

try:
    from greenlet import getcurrent as get_ident
except ImportError:
    try:
        from thread import get_ident
    except ImportError:
        from _thread import get_ident


class CameraEvent(object):
    """An Event-like class that signals all active clients when a new frame is
    available.
    """
    def __init__(self):
        self.events = {}

    def wait(self):
        """Invoked from each client's thread to wait for the next frame."""
        ident = get_ident()
        if ident not in self.events:
            # this is a new client
            # add an entry for it in the self.events dict
            # each entry has two elements, a threading.Event() and a timestamp
            self.events[ident] = [threading.Event(), time.time()]
        return self.events[ident][0].wait()

    def set(self):
        """Invoked by the camera thread when a new frame is available."""
        now = time.time()
        remove = None
        for ident, event in self.events.items():
            if not event[0].isSet():
                # if this client's event is not set, then set it
                # also update the last set timestamp to now
                event[0].set()
                event[1] = now
            else:
                # if the client's event is already set, it means the client
                # did not process a previous frame
                # if the event stays set for more than 5 seconds, then assume
                # the client is gone and remove it
                if now - event[1] > 5:
                    remove = ident
        if remove:
            del self.events[remove]

    def clear(self):
        """Invoked from each client's thread after a frame was processed."""
        self.events[get_ident()][0].clear()


class BaseCamera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera
    event = CameraEvent()
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 
                                        "haarcascade_frontalface_default.xml")

    def __init__(self):
        """Start the background camera thread if it isn't running yet."""
        if BaseCamera.thread is None:
            BaseCamera.last_access = time.time()

            # start background frame thread
            BaseCamera.thread = threading.Thread(target=self._thread)
            BaseCamera.thread.start()

            # wait until frames are available
            while self.get_frame() is None:
                time.sleep(0)

    def get_frame(self):
        """Return the current camera frame. Used in flask view."""
        BaseCamera.last_access = time.time()

        # wait for a signal from the camera thread
        BaseCamera.event.wait()
        BaseCamera.event.clear()

        return BaseCamera.frame

    @staticmethod
    def frames():
        """"Generator that returns frames from the camera."""
        raise RuntimeError('Must be implemented by subclasses.')

    @classmethod
    def _thread(cls):
        """Camera background thread."""
        print('Starting camera thread.')
        frames_iterator = cls.frames()
        for frame in frames_iterator:
            # TODO: process frame
            frame = Sudoku.process_frame(frame)
            frame = frame[...,::-1]  # RGB to BRG
            frame = cv.imencode('.jpg', frame)[1].tobytes()
            BaseCamera.frame = frame
            BaseCamera.event.set()  # send signal to clients
            time.sleep(0)

            # if there hasn't been any client asking for frames in
            # the last 10 seconds then stop the thread
            if time.time() - BaseCamera.last_access > 10:
                frames_iterator.close()
                print('Stopping camera thread due to inactivity.')
                break
        BaseCamera.thread = None


class PiCamera(BaseCamera):
    @staticmethod
    def frames():
        with picamera.PiCamera() as camera:
            # let camera warm up
            time.sleep(2)

            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, 'jpeg',
                                               use_video_port=True):
                # return current frame
                stream.seek(0)
                frame = stream.read()
                # BGR to RGB
                yield frame[...,::-1]

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()


class OpenCVCamera(BaseCamera):
    video_source = 0

    def __init__(self):
        if os.getenv('OPENCV_CAMERA_SOURCE'):
            OpenCVCamera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(OpenCVCamera, self).__init__()

    @staticmethod
    def set_video_source(source):
        OpenCVCamera.video_source = source

    @staticmethod
    def frames():
        camera = cv.VideoCapture(OpenCVCamera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame (array type)
            _, frame = camera.read()

            # BGR to RGB
            yield frame[...,::-1]


class EmuCamera(BaseCamera):
    """An emulated camera implementation that streams a repeated sequence of
    files."""
    img_dir = Path(__file__).parent / 'img'
    imgs = []
    for f in img_dir.iterdir():
        img = skimage.io.imread(f)
        img = reduce_image_size(img)

        if img.shape[0] > img.shape[1]:
            img = skimage.transform.rotate(img, 90, resize=True, 
                                           preserve_range=True)
            img = img.astype(np.int64)
        
        imgs.append(img)

    @staticmethod
    def frames():
        i = 0
        while True:
            time.sleep(3)
            frame = EmuCamera.imgs[i]
            yield frame

            if i == len(EmuCamera.imgs) - 1:
                i = 0
            else:
                i += 1
