import itertools
from pathlib import Path
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, img_as_ubyte, io, morphology
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, square
from skimage.transform import (ProjectiveTransform, hough_line,
                               hough_line_peaks, resize, warp)


def byte_string_to_array(img):
    img = np.frombuffer(img, np.uint8)
    img = cv.imdecode(img, cv.IMREAD_COLOR)
    return img[...,::-1]  # RGB to BGR


def load_image(image_path, resize_img=True, grayscale_img=False):
    image = io.imread(image_path)

    if resize_img:
        image = reduce_image_size(image)
    if grayscale_img:
        image = rgb2gray(image)

    return image


def reduce_image_size(image, sm_edge_px=480):
    """Resize image such that shortest edge is length sm_edge_px"""
    if image.shape[1] < image.shape[0]:
        to_shape = (image.shape[0] // (image.shape[1] / sm_edge_px), 
                    sm_edge_px)
    else:
        to_shape = (sm_edge_px, 
                    image.shape[1] // (image.shape[0] / sm_edge_px))
    return resize(image, to_shape, preserve_range=True)


def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return None


def detect_board(image, plot=False):
    gray = rgb2gray(image)

    bw = gray > (threshold_otsu(gray))  # to black & white
    bw = opening(bw, square(9))  # remove isolated white spots
    filled = ndi.binary_fill_holes(bw) 

    # only keep shapes larger than 1/4 of the image area
    cleaned = morphology.remove_small_objects(
        filled, image.shape[0] * image.shape[1] / 4 
    )

    edge = canny(cleaned)  # get edges of large shape
    
    # get straight lines
    h, theta, d = hough_line(edge)
    _, angles, dists = hough_line_peaks(
        hspace=h, angles=theta, dists=d, num_peaks=4, 
        threshold=0.5
    )

    lines = []
    for angle, C in zip(angles, dists):
        # Ax + By = C
        A = np.cos(angle)
        B = np.sin(angle)
        lines.append((A, B, C))

    corners = []
    for L1, L2 in itertools.combinations(lines, 2):
        pt = intersection(L1, L2)
        if not pt:
            continue
        conditions = [  # must intersect within 50px of the image border
             pt[0] > -50,
             pt[1] > -50, 
             pt[0] < image.shape[1] + 50,
             pt[1] < image.shape[0] + 50, 
        ]
        if all(conditions):
            corners.append(pt)

    corners = np.array(sort_points(corners))

    if plot:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
        ax[0][0].imshow(bw, cmap=plt.cm.gray)
        ax[0][1].imshow(cleaned, cmap=plt.cm.gray)
        ax[1][0].imshow(edge, cmap=plt.cm.gray)

        ax[1][1].imshow(image, cmap=plt.cm.gray)
        ax[1][1].plot(corners[:, 0], corners[:, 1], '.r')

        for angle, dist in zip(angles, dists):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
            ax[1][1].plot((0, image.shape[1]), (y0, y1), '-r')

        ax[1][1].set_xlim((0, image.shape[1]))
        ax[1][1].set_ylim((image.shape[0], 0))

        plt.tight_layout()
        plt.show()

    return corners


def sort_points(corners):
    """Sorts points into top left, bottom left, bottom right, top right"""
    assert len(corners) == 4

    def average(ls):
        return sum(ls) / len(ls)

    def is_top_left(pt, centroid):
        return pt[0] < centroid[0] and pt[1] < centroid[1]

    def is_bottom_left(pt, centroid):
        return pt[0] < centroid[0] and pt[1] > centroid[1]

    def is_bottom_right(pt, centroid):
        return pt[0] > centroid[0] and pt[1] > centroid[1]

    def is_top_right(pt, centroid):
        return pt[0] > centroid[0] and pt[1] < centroid[1]

    centroid = (average([t[0] for t in corners]),
                average([t[1] for t in corners]))

    pts_sorted = []
    for check in [is_top_left, is_bottom_left, is_bottom_right, is_top_right]:
        for i in range(len(corners)):
            if check(corners[i], centroid):
                pts_sorted.append(corners.pop(i))
                break
            elif i == len(corners)-1:  # if no match by end of the list
                raise ValueError(f'Could not find point for: {check.__name__}')

    return pts_sorted


def ball_loc(image, x, y, sudoku_size=9):
    """Given a square board image, returns ball array by position"""
    ball_size = image.shape[0] // sudoku_size

    ball = image[
        ball_size * y : ball_size * y + ball_size, 
        ball_size * x : ball_size * x + ball_size, 
    ]

    return ball


def perspective_fix(brd_image, brd_pts):
    n_px = 480
    src = np.array([[   0,    0],
                    [   0, n_px], 
                    [n_px, n_px],
                    [n_px,    0]])
    dst = np.array(sort_points(brd_pts))
    tform = ProjectiveTransform()
    tform.estimate(src, dst)
    brd_image = warp(brd_image, tform, output_shape=(n_px, n_px))

    crop = 40
    brd_image = brd_image[crop:-crop, crop:-crop]
    brd_image = resize(brd_image, (99, 99))

    return brd_image


def get_balls(image, sudoku_size=9):
    """Processes a square board image into a matrix with rows of ball data"""
    # convert each ball shape to 1D array
    balls = []
    for y in range(sudoku_size):
        for x in range(sudoku_size):
            ball = ball_loc(image, x, y, sudoku_size=sudoku_size)
            balls.append(ball.flatten())
            
    return np.vstack(balls)
