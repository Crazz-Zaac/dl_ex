import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import unittest
from src_to_implement.NumpyTests import *
from numpy import ndarray
from dataclasses import dataclass
import os
from typing import Tuple, List


class Checker:
    def __init__(self, resolution: int, tile_size: int):
        # check if resolution is evenly divisible by tile size
        # to avoid truncated checkerboard
        if resolution % (2 * tile_size) != 0:
            raise ValueError("Resolution value must be equally divisible")
        self.tile_size = tile_size
        self.resolution = resolution

    def draw(self) -> ndarray:
        # binary checkerboard with 1s and 0s pattern
        number_checkerboard = (
            np.indices(
                (self.resolution // self.tile_size, self.resolution // self.tile_size)
            ).sum(axis=0)
            % 2
        )

        binary_checkerboard = np.kron(
            number_checkerboard, np.ones((self.tile_size, self.tile_size), dtype=int)
        )

        self.output = binary_checkerboard
        return self.output.copy()

    def show(self):
        self.output = self.draw()
        plt.imshow(self.output, cmap="gray")  # , interpolation='nearest')
        plt.axis("off")
        plt.title("Output")
        plt.show()


class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple) -> None:
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self) -> ndarray:
        # creating grid points
        along_x_coordinates = np.arange(self.resolution)
        along_y_coordinates = np.arange(self.resolution)
        xx, yy = np.meshgrid(along_x_coordinates, along_y_coordinates)

        # calculating distance from center, the position tuple
        distance_from_position = np.sqrt(
            (xx - self.position[0]) ** 2 + (yy - self.position[1]) ** 2
        )

        # creating circle
        binary_circle = (distance_from_position <= self.radius).astype(int)

        self.output = binary_circle
        return self.output.copy()

    def show(self):
        self.output = self.draw()
        plt.imshow(self.output, cmap="gray")  # , interpolation='nearest')
        plt.axis("off")
        plt.title("Output")
        plt.show()


class Spectrum:
    def __init__(self, resolution: int) -> None:
        self.resolution = resolution
        # initialize the output array for RGB channels of (resolution x resolution) size
        self.output = np.zeros((self.resolution, self.resolution, 3))

    def draw(self) -> ndarray:
        # create a grid of resolution x resolution within the range of 0 to 1
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)

        # creating the meshgrid of RGB channels
        red_channel, green_channel = np.meshgrid(x, y)
        blue_channel = 1 - red_channel

        # create the 3 channels of the output array
        self.output[:, :, 0] = red_channel
        self.output[:, :, 1] = green_channel
        self.output[:, :, 2] = blue_channel

        print("value of x", x)
        print("value of y", y)
        print("\n\nvalue of red_channel:\n", red_channel)
        print("value of green_channel:\n", green_channel)
        print("value of blue_channel:\n", blue_channel)

        return self.output.copy()

    def show(self):
        self.output = self.draw()
        plt.imshow(self.output)
        plt.axis("off")
        plt.title("Output")
        plt.show()


