import os
from pathlib import Path

import cv2 as cv

from .base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 
                                        "haarcascade_frontalface_default.xml")

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, frame = camera.read()

            grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = Camera.face_cascade.detectMultiScale(grayscale_frame)

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # encode as a jpeg image and return it
            yield cv.imencode('.jpg', frame)[1].tobytes()
