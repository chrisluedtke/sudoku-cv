# Sudoku Computer Vision Solver

A deep neural network trained from scratch. > 99% accurate in various lighting conditions.

![](img/demo.gif)

## Setup

1. Clone this repository (fork it first if you want to deploy to Heroku):
    ```
    git clone https://github.com/chrisluedtke/sudoku-cv.git
    ```
2. Within project repository, create a virtual environment:
    ```
    python -m venv env
    ```
3. Activate the environment:
    
    Windows:
    ```
    env\Scripts\activate
    ```
4. `pip install --user --upgrade pip`
5. Install requirements:
    ```
    pip install -r requirements-opencv.txt
    ```
    or
    ```
    pip install -r requirements-pi.txt
    ```
6. Create a `.env` with contents:
    ```
    FLASK_APP=app:APP
    FLASK_ENV=development
    CAMERA=opencv
    ```
7. Run the web app (the database will be initialized automatically):
    ```
    flask run
    ```
8. Navigate to the locally served page, typically `http://localhost:5000/`
7. *Optional:* Create an ipython kernel to use Jupyter Notebook with this environment (see [documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)).
    ```
    ipython kernel install --user --name=sudoku
    ```
    * You may also need to copy the `.dll` files from `envLib/site-packages/pywin32_system32` to `env/Lib/site-packages/win32`