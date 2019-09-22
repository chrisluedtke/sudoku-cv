import os

from flask import Flask, Response, render_template, request

from .config import Config
from .sudoku import Sudoku
from .sudoku_cv import detect_board, byte_string_to_array

if os.getenv('CAMERA') == 'pi':
    from .camera import PiCamera as Camera
elif os.getenv('CAMERA') == 'opencv':
    from .camera import OpenCVCamera as Camera
else:
    from .camera import EmuCamera as Camera


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    app = Flask(__name__)

    @app.route('/', methods=('GET', 'POST'))
    def index():
        """Video streaming home page."""
        if request.method == 'POST':
            camera = Camera()
            frame = camera.get_frame()
            frame = byte_string_to_array(frame)
            board_coords = detect_board(frame)
            Sudoku.set_board_coords(board_coords)
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route. Referenced src attribute of an img tag."""

        def gen(camera):
            """Video streaming generator function."""
            while True:
                frame = camera.get_frame()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        camera = Camera()

        return Response(gen(camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    return app
