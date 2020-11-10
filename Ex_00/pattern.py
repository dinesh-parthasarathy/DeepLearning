from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        nt = (self.resolution // self.tile_size)
        if nt % 2 != 0:
            print('Truncated checkerboard pattern, use other input values ')
            return
        n = nt * self.tile_size
        self.output = np.zeros((n, n), dtype=int)

    def draw(self):
        nt = self.resolution // self.tile_size
        if nt % 2 != 0:
            print('Truncated checkerboard pattern, use other input values ')
            return
        t0 = np.zeros((self.tile_size, self.tile_size))
        t1 = np.ones((self.tile_size, self.tile_size))
        tbig = np.concatenate((np.concatenate((t0, t1), axis=1), np.concatenate((t1, t0), axis=1)), axis=0)
        self.output = np.tile(tbig, (nt // 2, nt // 2))

        return deepcopy(self.output)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, resolution, radius: int, position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = np.ones((resolution, resolution), dtype=int)

    def draw(self):
        y, x = np.ogrid[0:self.resolution, 0:self.resolution]
        self.output = ((y - self.position[1]) ** 2 + (x - self.position[0]) ** 2) < self.radius ** 2
        return deepcopy(self.output)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.array((resolution, resolution, 3), dtype=float)

    def draw(self):
        r = np.tile(np.linspace(0, 1, self.resolution), (self.resolution, 1))
        g = np.rot90(r, 3)
        b = np.rot90(g, 3)
        self.output = np.dstack((r, g, b))
        return deepcopy(self.output)

    def show(self):
        plt.imshow(self.output)
        plt.show()
