import os

from flask import Flask, Response, render_template

from .config import Config

if os.environ.get('CAMERA') == 'pi':
    from .camera_pi import Camera
else:
    from .camera_opencv import Camera


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app = Flask(__name__)

    @app.route('/')
    def index():
        """Video streaming home page."""
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route. Referenced src attribute of an img tag."""
        return Response(gen(Camera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
