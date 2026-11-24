import unittest

import numpy as np

from scripts import grqa

from numba import cuda
from scipy.spatial.distance import pdist


class TestOnesSquared(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.img = np.ones((10, 10))
        cls.epsilon = 1

        cls.mass_distribution_expected = np.array([0, 10000, 10000, 6561, 4096, 10000, 1296, 2401, 4096, 6561, 10000])
        cls.mass_distribution_reverse_expected = np.zeros(11)
        cls.mass_distribution_z2_expected = np.array([0, 10000, 160000, 531441, 1048576, 6250000, 1679616, 5764801, 16777216, 43046721, 100000000])
        cls.mass_distribution_reverse_z2_expected = np.zeros(11)
        cls.nr_boxes_expected = np.array([0, 10000, 625, 81, 16, 16, 1, 1, 1, 1, 1])


    def test_max_max_box_size(self):

        max_box_size = 10

        args = [{}, {"memory_efficient": True}, {}]
        for i, func in enumerate([grqa.mass_distribution_fast, grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, self.mass_distribution_expected)
            np.testing.assert_array_equal(mass_distribution_reverse, self.mass_distribution_reverse_expected)
            np.testing.assert_array_equal(mass_distribution_z2, self.mass_distribution_z2_expected)
            np.testing.assert_array_equal(mass_distribution_reverse_z2, self.mass_distribution_reverse_z2_expected)
            np.testing.assert_array_equal(nr_boxes, self.nr_boxes_expected)


    def test_small_max_box_size(self):

        max_box_size = 5

        args = [{}, {"memory_efficient": True}, {}]
        for i, func in enumerate(
                [grqa.mass_distribution_fast, grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(
                self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, self.mass_distribution_expected[:6])
            np.testing.assert_array_equal(mass_distribution_reverse, self.mass_distribution_reverse_expected[:6])
            np.testing.assert_array_equal(mass_distribution_z2, self.mass_distribution_z2_expected[:6])
            np.testing.assert_array_equal(mass_distribution_reverse_z2, self.mass_distribution_reverse_z2_expected[:6])
            np.testing.assert_array_equal(nr_boxes, self.nr_boxes_expected[:6])


    def test_too_large_max_box_size(self):

        max_box_size = 20

        args = [{}, {"memory_efficient": True}, {}]
        for i, func in enumerate(
                [grqa.mass_distribution_fast, grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(
                self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, np.hstack([self.mass_distribution_expected, np.zeros(10)]))
            np.testing.assert_array_equal(mass_distribution_reverse, np.hstack([self.mass_distribution_reverse_expected, np.zeros(10)]))
            np.testing.assert_array_equal(mass_distribution_z2, np.hstack([self.mass_distribution_z2_expected, np.zeros(10)]))
            np.testing.assert_array_equal(mass_distribution_reverse_z2, np.hstack([self.mass_distribution_reverse_z2_expected, np.zeros(10)]))
            np.testing.assert_array_equal(nr_boxes, np.hstack([self.nr_boxes_expected, np.zeros(10)]))


    def test_store_box_size(self):

        if not cuda.is_available():
            print("No GPU available. Test can't be run.")
        else:
            # from this example we would expect that there are mass values for the box sizes 1...5 (until max_box_size)
            # and then additionally for 6 and 9 (integer multiples of the stored box size 3),
            # but NO values for 7, 8 and 10

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = grqa.mass_distribution_gpu_large(
                self.img,
                self.epsilon,
                max_box_size=5,
                box_size_store=3,
                max_box_size_large=10,
            )

            # compare the mass values
            np.testing.assert_array_equal(mass_distribution[[0, 1, 2, 3, 4, 5, 6, 9]], self.mass_distribution_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(mass_distribution_reverse[[0, 1, 2, 3, 4, 5, 6, 9]], self.mass_distribution_reverse_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(mass_distribution_z2[[0, 1, 2, 3, 4, 5, 6, 9]], self.mass_distribution_z2_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(mass_distribution_reverse_z2[[0, 1, 2, 3, 4, 5, 6, 9]], self.mass_distribution_reverse_z2_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(nr_boxes[[0, 1, 2, 3, 4, 5, 6, 9]], self.nr_boxes_expected[[0, 1, 2, 3, 4, 5, 6, 9]])

            # test if the expected values are 0
            np.testing.assert_array_equal(mass_distribution[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(mass_distribution_reverse[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(mass_distribution_z2[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(mass_distribution_reverse_z2[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(nr_boxes[[7, 8, 10]], np.zeros(3))


    def test_lacunarity(self):

        lac_expected = np.ones(11) * np.nan
        lac = grqa.lacunarity(self.mass_distribution_expected, self.mass_distribution_reverse_expected, self.mass_distribution_z2_expected, self.mass_distribution_reverse_z2_expected, self.nr_boxes_expected)

        np.testing.assert_array_equal(lac, lac_expected)


class TestOnesRectangle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.img = np.ones((10, 5))
        cls.epsilon = 1

        cls.mass_distribution_expected = np.array([ 0, 2500, 1600, 729, 1024, 2500])
        cls.mass_distribution_reverse_expected = np.zeros(6)
        cls.mass_distribution_z2_expected = np.array([ 0, 2500, 25600, 59049, 262144, 1562500])
        cls.mass_distribution_reverse_z2_expected = np.zeros(6)
        cls.nr_boxes_expected = np.array([ 0, 2500, 100, 9, 4, 4])


    def test_max_max_box_size(self):

        max_box_size = 5

        args = [{}, {"memory_efficient": True}, {}]
        for i, func in enumerate([grqa.mass_distribution_fast, grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, self.mass_distribution_expected)
            np.testing.assert_array_equal(mass_distribution_reverse, self.mass_distribution_reverse_expected)
            np.testing.assert_array_equal(mass_distribution_z2, self.mass_distribution_z2_expected)
            np.testing.assert_array_equal(mass_distribution_reverse_z2, self.mass_distribution_reverse_z2_expected)
            np.testing.assert_array_equal(nr_boxes, self.nr_boxes_expected)


    def test_small_max_box_size(self):

        max_box_size = 3

        args = [{}, {"memory_efficient": True}, {}]
        for i, func in enumerate(
                [grqa.mass_distribution_fast, grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(
                self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, self.mass_distribution_expected[:4])
            np.testing.assert_array_equal(mass_distribution_reverse, self.mass_distribution_reverse_expected[:4])
            np.testing.assert_array_equal(mass_distribution_z2, self.mass_distribution_z2_expected[:4])
            np.testing.assert_array_equal(mass_distribution_reverse_z2, self.mass_distribution_reverse_z2_expected[:4])
            np.testing.assert_array_equal(nr_boxes, self.nr_boxes_expected[:4])


    def test_too_large_max_box_size(self):

        max_box_size = 20

        args = [{}, {"memory_efficient": True}, {}]
        for i, func in enumerate(
                [grqa.mass_distribution_fast, grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(
                self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, np.hstack([self.mass_distribution_expected, np.zeros(15)]))
            np.testing.assert_array_equal(mass_distribution_reverse,
                                          np.hstack([self.mass_distribution_reverse_expected, np.zeros(15)]))
            np.testing.assert_array_equal(mass_distribution_z2,
                                          np.hstack([self.mass_distribution_z2_expected, np.zeros(15)]))
            np.testing.assert_array_equal(mass_distribution_reverse_z2,
                                          np.hstack([self.mass_distribution_reverse_z2_expected, np.zeros(15)]))
            np.testing.assert_array_equal(nr_boxes, np.hstack([self.nr_boxes_expected, np.zeros(15)]))


    def test_in_between_max_box_size(self):

        max_box_size = 8

        args = [{}, {"memory_efficient": True}, {}]
        for i, func in enumerate(
                [grqa.mass_distribution_fast, grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(
                self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, np.hstack([self.mass_distribution_expected, np.zeros(3)]))
            np.testing.assert_array_equal(mass_distribution_reverse, np.hstack([self.mass_distribution_reverse_expected, np.zeros(3)]))
            np.testing.assert_array_equal(mass_distribution_z2, np.hstack([self.mass_distribution_z2_expected, np.zeros(3)]))
            np.testing.assert_array_equal(mass_distribution_reverse_z2, np.hstack([self.mass_distribution_reverse_z2_expected, np.zeros(3)]))
            np.testing.assert_array_equal(nr_boxes, np.hstack([self.nr_boxes_expected, np.zeros(3)]))


    def test_store_box_size(self):

        if not cuda.is_available():
            print("No GPU available. Test can't be run.")
        else:
            # from this example we would expect that there are mass values for the box sizes 1...5 (until max_box_size)
            # and then additionally for 6 and 9 (integer multiples of the stored box size 3),
            # but NO values for 7, 8 and 10

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = grqa.mass_distribution_gpu_large(
                self.img,
                self.epsilon,
                max_box_size=5,
                box_size_store=3,
                max_box_size_large=10,
            )

            # compare the mass values
            np.testing.assert_array_equal(mass_distribution, np.hstack([self.mass_distribution_expected, np.zeros(5)]))
            np.testing.assert_array_equal(mass_distribution_reverse, np.hstack([self.mass_distribution_reverse_expected, np.zeros(5)]))
            np.testing.assert_array_equal(mass_distribution_z2, np.hstack([self.mass_distribution_z2_expected, np.zeros(5)]))
            np.testing.assert_array_equal(mass_distribution_reverse_z2, np.hstack([self.mass_distribution_reverse_z2_expected, np.zeros(5)]))
            np.testing.assert_array_equal(nr_boxes, np.hstack([self.nr_boxes_expected, np.zeros(5)]))


    def test_lacunarity(self):

        lac_expected = np.ones(6) * np.nan
        lac = grqa.lacunarity(self.mass_distribution_expected, self.mass_distribution_reverse_expected, self.mass_distribution_z2_expected, self.mass_distribution_reverse_z2_expected, self.nr_boxes_expected)

        np.testing.assert_array_equal(lac, lac_expected)


class TestPeriodicImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        size = 25
        x = np.linspace(0, 8 * np.pi, size)
        y = np.linspace(0, 8 * np.pi, size)  # for checkerboard like pattern
        X, Y = np.meshgrid(x, y)

        # Create a periodic pattern using sine or cosine functions
        cls.img = np.sin(X) + np.cos(Y)
        cls.epsilon = np.quantile(
        pdist(
            np.c_[cls.img.flatten(), cls.img.flatten()],
            metric="euclidean",
        ),
        0.2)

        # transform to float32 since this is also the datatype the GPU works with
        cls.img = cls.img.astype(np.float32)
        cls.epsilon = np.float32(cls.epsilon)

        cls.mass_distribution_expected = np.array([ 0, 75897, 65536, 65536, 65536, 75897, 65536, 38491, 65536, 20736, 31288])
        cls.mass_distribution_reverse_expected = np.array([ 0, 314728, 266240, 266240, 266240, 314728, 266240, 155990, 266240, 84240, 128712])
        cls.mass_distribution_z2_expected = np.array([0, 75897, 335872, 1157120, 3409408, 9247505, 16777216, 18318355, 53155232, 26905320, 61245200])
        cls.mass_distribution_reverse_z2_expected = np.array([ 0, 314728, 3547136, 17414144, 54789632, 158516880, 276889600, 300433454, 875238816, 443555064, 1035485200])
        cls.nr_boxes_expected = np.array([0, 390625, 20736, 4096, 1296, 625, 256, 81, 81, 16, 16])

    def test_max_box_size(self):

        max_box_size = 10

        args = [{}, {}]
        for i, func in enumerate(
                [grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(
                self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, self.mass_distribution_expected)
            np.testing.assert_array_equal(mass_distribution_reverse, self.mass_distribution_reverse_expected)
            np.testing.assert_array_equal(mass_distribution_z2, self.mass_distribution_z2_expected)
            np.testing.assert_array_equal(mass_distribution_reverse_z2, self.mass_distribution_reverse_z2_expected)
            np.testing.assert_array_equal(nr_boxes, self.nr_boxes_expected)


    def test_small_max_box_size(self):

        max_box_size = 5

        args = [{}, {}]
        for i, func in enumerate(
                [grqa.mass_distribution_fast, grqa.mass_distribution_gpu]):

            if func == grqa.mass_distribution_gpu and not cuda.is_available():
                print("No GPU available. Test can't be run.")
                continue

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = func(
                self.img, self.epsilon, max_box_size=max_box_size, **args[i])
            np.testing.assert_array_equal(mass_distribution, self.mass_distribution_expected[:6])
            np.testing.assert_array_equal(mass_distribution_reverse, self.mass_distribution_reverse_expected[:6])
            np.testing.assert_array_equal(mass_distribution_z2, self.mass_distribution_z2_expected[:6])
            np.testing.assert_array_equal(mass_distribution_reverse_z2, self.mass_distribution_reverse_z2_expected[:6])
            np.testing.assert_array_equal(nr_boxes, self.nr_boxes_expected[:6])


    def test_store_box_size(self):

        if not cuda.is_available():
            print("No GPU available. Test can't be run.")
        else:
            # from this example we would expect that there are mass values for the box sizes 1...5 (until max_box_size)
            # and then additionally for 6 and 9 (integer multiples of the stored box size 3),
            # but NO values for 7, 8 and 10

            mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = grqa.mass_distribution_gpu_large(
                self.img,
                self.epsilon,
                max_box_size=5,
                box_size_store=3,
                max_box_size_large=10,
            )

            # compare the mass values
            np.testing.assert_array_equal(mass_distribution[[0, 1, 2, 3, 4, 5, 6, 9]],
                                          self.mass_distribution_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(mass_distribution_reverse[[0, 1, 2, 3, 4, 5, 6, 9]],
                                          self.mass_distribution_reverse_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(mass_distribution_z2[[0, 1, 2, 3, 4, 5, 6, 9]],
                                          self.mass_distribution_z2_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(mass_distribution_reverse_z2[[0, 1, 2, 3, 4, 5, 6, 9]],
                                          self.mass_distribution_reverse_z2_expected[[0, 1, 2, 3, 4, 5, 6, 9]])
            np.testing.assert_array_equal(nr_boxes[[0, 1, 2, 3, 4, 5, 6, 9]],
                                          self.nr_boxes_expected[[0, 1, 2, 3, 4, 5, 6, 9]])

            # test if the expected values are 0
            np.testing.assert_array_equal(mass_distribution[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(mass_distribution_reverse[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(mass_distribution_z2[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(mass_distribution_reverse_z2[[7, 8, 10]], np.zeros(3))
            np.testing.assert_array_equal(nr_boxes[[7, 8, 10]], np.zeros(3))


    def test_lacunarity(self):

        lac_expected = np.array([np.nan, 1., 0.419614, 0.1000384, 0.02971949, 0.00354068, 0., 0.00159432, 0.00261304, 0.00124037, 0.00106383])
        lac = grqa.lacunarity(self.mass_distribution_expected, self.mass_distribution_reverse_expected,
                              self.mass_distribution_z2_expected, self.mass_distribution_reverse_z2_expected,
                              self.nr_boxes_expected)

        np.testing.assert_allclose(lac, lac_expected, atol=5e-09)
