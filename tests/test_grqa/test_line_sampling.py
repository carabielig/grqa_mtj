import unittest

import numpy as np

from scripts import grqa

from numba import cuda

"""
The fist test shows that the slow approach leads to the correct results, with and without rotation.
(Here, a test for overlapping surfaces could be added)

In the second test the results of the slow cpu approach will be compared to the fast one to check
if they return the same results.

In the third test a rectangular image will be tested with the same components as the squared image, so that
the distributions should be the same, except for the main diagonal HS (and some other minor differences). 
Therefore the results are will be compared to the results from the second test.

Finally, there will be a test for the GRQA measures for known distributions.
"""


class TestArtificialRP(unittest.TestCase):

    def test_artificial_rp(self):
        """
        Test an artificial RP with one (known) vertical and one diagonal HS.
        This is a simple example to test the detection of vertical and diagonal HS with and
        without rotation with the correct indices.
        """

        def add_diag_hs(y1, x1, y2, x2, length, rp, k=[1, 1, 1, 1]):
            a, b, c, d = k
            for k1 in range(length):
                for k2 in range(length):
                    rp[y1 + a * k1, x1 + b * k2, y2 + c * k1, x2 + d * k2] = 1
            return rp

        def add_vert_hs(y1, x1, y2, x2, length, rp, k=[1, 1]):
            a, b = k
            for k1 in range(length):
                for k2 in range(length):
                    rp[y1, x1, y2 + a * k1, x2 + b * k2] = 1
            return rp

        rp = np.zeros((20, 20, 20, 20))

        rp = add_diag_hs(1, 2, 1, 2, 5, rp)
        rp = add_vert_hs(8, 10, 8, 10, 10, rp)

        rp_sum1, ind1, len1, dirs1, dir_fac1 = grqa.diag_hs_patch_sampling(
            rp, return_distribution=False
        )
        np.testing.assert_array_equal(ind1, np.array([[1, 2, 1, 2]]))
        np.testing.assert_array_equal(len1, np.array([5]))
        np.testing.assert_array_equal(dirs1, np.array([[1, 1, 1, 1]]))
        np.testing.assert_equal(dir_fac1, 1)

        rp_sum2, ind2, len2, dirs2, dir_fac2 = grqa.diag_hs_patch_sampling(
            rp, rot=True, return_distribution=False
        )
        np.testing.assert_array_equal(
            ind2, np.array([[1, 2, 1, 2], [5, 2, 5, 2], [5, 6, 5, 6], [1, 6, 1, 6]])
        )
        np.testing.assert_array_equal(len2, np.array([5, 5, 5, 5]))
        np.testing.assert_array_equal(
            dirs2,
            np.array([[1, 1, 1, 1], [-1, 1, -1, 1], [-1, -1, -1, -1], [1, -1, 1, -1]]),
        )
        np.testing.assert_equal(dir_fac2, 16)

        rp_sum3, ind3, len3, dirs3, dir_fac3 = grqa.diag_hs_line_sampling(
            rp, return_distribution=False
        )
        np.testing.assert_array_equal(
            ind3,
            np.array(
                [
                    [1, 2, 1, 2],
                    [1, 3, 1, 3],
                    [1, 4, 1, 4],
                    [1, 5, 1, 5],
                    [1, 6, 1, 6],
                    [1, 2, 1, 2],
                    [1, 3, 1, 3],
                    [1, 4, 1, 4],
                    [1, 5, 1, 5],
                    [2, 2, 2, 2],
                    [3, 2, 3, 2],
                    [4, 2, 4, 2],
                ]
            ),
        )
        np.testing.assert_array_equal(
            len3, np.array([5, 5, 5, 5, 5, 5, 4, 3, 2, 4, 3, 2])
        )
        np.testing.assert_array_equal(
            dirs3,
            np.array(
                [
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ]
            ),
        )
        np.testing.assert_equal(dir_fac3, 4)

        rp_sum4, ind4, len4, dirs4, dir_fac4 = grqa.diag_hs_line_sampling(
            rp, rot=True, return_distribution=False
        )
        np.testing.assert_array_equal(
            ind4,
            np.array(
                [
                    [1, 2, 1, 2],
                    [1, 3, 1, 3],
                    [1, 4, 1, 4],
                    [1, 5, 1, 5],
                    [1, 6, 1, 6],
                    [1, 2, 1, 2],
                    [1, 3, 1, 3],
                    [1, 4, 1, 4],
                    [1, 5, 1, 5],
                    [2, 2, 2, 2],
                    [3, 2, 3, 2],
                    [4, 2, 4, 2],
                    [1, 2, 1, 2],
                    [2, 2, 2, 2],
                    [3, 2, 3, 2],
                    [4, 2, 4, 2],
                    [5, 2, 5, 2],
                    [2, 2, 2, 2],
                    [3, 2, 3, 2],
                    [4, 2, 4, 2],
                    [5, 2, 5, 2],
                    [5, 3, 5, 3],
                    [5, 4, 5, 4],
                    [5, 5, 5, 5],
                    [5, 2, 5, 2],
                    [5, 3, 5, 3],
                    [5, 4, 5, 4],
                    [5, 5, 5, 5],
                    [5, 6, 5, 6],
                    [2, 6, 2, 6],
                    [3, 6, 3, 6],
                    [4, 6, 4, 6],
                    [5, 3, 5, 3],
                    [5, 4, 5, 4],
                    [5, 5, 5, 5],
                    [5, 6, 5, 6],
                    [1, 6, 1, 6],
                    [2, 6, 2, 6],
                    [3, 6, 3, 6],
                    [4, 6, 4, 6],
                    [5, 6, 5, 6],
                    [1, 3, 1, 3],
                    [1, 4, 1, 4],
                    [1, 5, 1, 5],
                    [1, 6, 1, 6],
                    [2, 6, 2, 6],
                    [3, 6, 3, 6],
                    [4, 6, 4, 6],
                ]
            ),
        )
        np.testing.assert_array_equal(
            len4,
            np.array(
                [
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    4,
                    3,
                    2,
                    4,
                    3,
                    2,
                    5,
                    5,
                    5,
                    5,
                    5,
                    2,
                    3,
                    4,
                    5,
                    4,
                    3,
                    2,
                    5,
                    5,
                    5,
                    5,
                    5,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    5,
                    5,
                    5,
                    5,
                    5,
                    5,
                    2,
                    3,
                    4,
                    5,
                    4,
                    3,
                    2,
                ]
            ),
        )
        np.testing.assert_array_equal(
            dirs4,
            np.array(
                [
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 1],
                    [0, 1, 0, 1],
                    [-1, 1, -1, 1],
                    [-1, 1, -1, 1],
                    [-1, 1, -1, 1],
                    [-1, 1, -1, 1],
                    [-1, 1, -1, 1],
                    [-1, 1, -1, 1],
                    [-1, 1, -1, 1],
                    [-1, 0, -1, 0],
                    [-1, 0, -1, 0],
                    [-1, 0, -1, 0],
                    [-1, 0, -1, 0],
                    [-1, 0, -1, 0],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [-1, -1, -1, -1],
                    [0, -1, 0, -1],
                    [0, -1, 0, -1],
                    [0, -1, 0, -1],
                    [0, -1, 0, -1],
                    [0, -1, 0, -1],
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                    [1, -1, 1, -1],
                ]
            ),
        )
        np.testing.assert_equal(dir_fac4, 64)

        rp_sum5, ind5, len5, dirs5, dir_fac5 = grqa.vert_hs_patch_sampling(
            rp, return_distribution=False
        )
        np.testing.assert_array_equal(ind5, np.array([[8, 10, 8, 10]]))
        np.testing.assert_array_equal(len5, np.array([10]))
        np.testing.assert_array_equal(dirs5, np.array([[1, 1]]))
        np.testing.assert_equal(dir_fac5, 1)

        rp_sum6, ind6, len6, dirs6, dir_fac6 = grqa.vert_hs_patch_sampling(
            rp, rot=True, return_distribution=False
        )
        np.testing.assert_array_equal(
            ind6,
            np.array(
                [[8, 10, 8, 10], [8, 10, 17, 10], [8, 10, 17, 19], [8, 10, 8, 19]]
            ),
        )
        np.testing.assert_array_equal(len6, np.array([10, 10, 10, 10]))
        np.testing.assert_array_equal(
            dirs6, np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        )
        np.testing.assert_equal(dir_fac6, 4)

        rp_sum7, ind7, len7, dirs7, dir_fac7 = grqa.vert_hs_line_sampling(
            rp, return_distribution=False
        )
        np.testing.assert_array_equal(
            ind7,
            np.array(
                [
                    [8, 10, 8, 10],
                    [8, 10, 8, 11],
                    [8, 10, 8, 12],
                    [8, 10, 8, 13],
                    [8, 10, 8, 14],
                    [8, 10, 8, 15],
                    [8, 10, 8, 16],
                    [8, 10, 8, 17],
                    [8, 10, 8, 18],
                    [8, 10, 8, 19],
                    [8, 10, 8, 10],
                    [8, 10, 8, 11],
                    [8, 10, 8, 12],
                    [8, 10, 8, 13],
                    [8, 10, 8, 14],
                    [8, 10, 8, 15],
                    [8, 10, 8, 16],
                    [8, 10, 8, 17],
                    [8, 10, 8, 18],
                    [8, 10, 9, 10],
                    [8, 10, 10, 10],
                    [8, 10, 11, 10],
                    [8, 10, 12, 10],
                    [8, 10, 13, 10],
                    [8, 10, 14, 10],
                    [8, 10, 15, 10],
                    [8, 10, 16, 10],
                ]
            ),
        )
        np.testing.assert_array_equal(
            len7,
            np.array(
                [
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                ]
            ),
        )
        np.testing.assert_array_equal(
            dirs7,
            np.array(
                [
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ]
            ),
        )
        np.testing.assert_equal(dir_fac7, 2)

        rp_sum8, ind8, len8, dirs8, dir_fac8 = grqa.vert_hs_line_sampling(
            rp, rot=True, return_distribution=False
        )
        np.testing.assert_array_equal(
            ind8,
            np.array(
                [
                    [8, 10, 8, 10],
                    [8, 10, 8, 11],
                    [8, 10, 8, 12],
                    [8, 10, 8, 13],
                    [8, 10, 8, 14],
                    [8, 10, 8, 15],
                    [8, 10, 8, 16],
                    [8, 10, 8, 17],
                    [8, 10, 8, 18],
                    [8, 10, 8, 19],
                    [8, 10, 8, 10],
                    [8, 10, 8, 11],
                    [8, 10, 8, 12],
                    [8, 10, 8, 13],
                    [8, 10, 8, 14],
                    [8, 10, 8, 15],
                    [8, 10, 8, 16],
                    [8, 10, 8, 17],
                    [8, 10, 8, 18],
                    [8, 10, 9, 10],
                    [8, 10, 10, 10],
                    [8, 10, 11, 10],
                    [8, 10, 12, 10],
                    [8, 10, 13, 10],
                    [8, 10, 14, 10],
                    [8, 10, 15, 10],
                    [8, 10, 16, 10],
                    [8, 10, 8, 10],
                    [8, 10, 9, 10],
                    [8, 10, 10, 10],
                    [8, 10, 11, 10],
                    [8, 10, 12, 10],
                    [8, 10, 13, 10],
                    [8, 10, 14, 10],
                    [8, 10, 15, 10],
                    [8, 10, 16, 10],
                    [8, 10, 17, 10],
                    [8, 10, 9, 10],
                    [8, 10, 10, 10],
                    [8, 10, 11, 10],
                    [8, 10, 12, 10],
                    [8, 10, 13, 10],
                    [8, 10, 14, 10],
                    [8, 10, 15, 10],
                    [8, 10, 16, 10],
                    [8, 10, 17, 10],
                    [8, 10, 17, 11],
                    [8, 10, 17, 12],
                    [8, 10, 17, 13],
                    [8, 10, 17, 14],
                    [8, 10, 17, 15],
                    [8, 10, 17, 16],
                    [8, 10, 17, 17],
                    [8, 10, 17, 18],
                    [8, 10, 17, 10],
                    [8, 10, 17, 11],
                    [8, 10, 17, 12],
                    [8, 10, 17, 13],
                    [8, 10, 17, 14],
                    [8, 10, 17, 15],
                    [8, 10, 17, 16],
                    [8, 10, 17, 17],
                    [8, 10, 17, 18],
                    [8, 10, 17, 19],
                    [8, 10, 9, 19],
                    [8, 10, 10, 19],
                    [8, 10, 11, 19],
                    [8, 10, 12, 19],
                    [8, 10, 13, 19],
                    [8, 10, 14, 19],
                    [8, 10, 15, 19],
                    [8, 10, 16, 19],
                    [8, 10, 17, 11],
                    [8, 10, 17, 12],
                    [8, 10, 17, 13],
                    [8, 10, 17, 14],
                    [8, 10, 17, 15],
                    [8, 10, 17, 16],
                    [8, 10, 17, 17],
                    [8, 10, 17, 18],
                    [8, 10, 17, 19],
                    [8, 10, 8, 19],
                    [8, 10, 9, 19],
                    [8, 10, 10, 19],
                    [8, 10, 11, 19],
                    [8, 10, 12, 19],
                    [8, 10, 13, 19],
                    [8, 10, 14, 19],
                    [8, 10, 15, 19],
                    [8, 10, 16, 19],
                    [8, 10, 17, 19],
                    [8, 10, 8, 11],
                    [8, 10, 8, 12],
                    [8, 10, 8, 13],
                    [8, 10, 8, 14],
                    [8, 10, 8, 15],
                    [8, 10, 8, 16],
                    [8, 10, 8, 17],
                    [8, 10, 8, 18],
                    [8, 10, 8, 19],
                    [8, 10, 9, 19],
                    [8, 10, 10, 19],
                    [8, 10, 11, 19],
                    [8, 10, 12, 19],
                    [8, 10, 13, 19],
                    [8, 10, 14, 19],
                    [8, 10, 15, 19],
                    [8, 10, 16, 19],
                ]
            ),
        )
        np.testing.assert_array_equal(
            len8,
            np.array(
                [
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                ]
            ),
        )
        np.testing.assert_array_equal(
            dirs8,
            np.array(
                [
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 1],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, 0],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [-1, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [0, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                    [1, -1],
                ]
            ),
        )
        np.testing.assert_equal(dir_fac8, 8)

        assert (
            rp_sum1
            == rp_sum2
            == rp_sum3
            == rp_sum4
            == rp_sum5
            == rp_sum6
            == rp_sum7
            == rp_sum8
        ), "RP sums do not match."

    # TODO: add artificial RP with overlapping HS


class TestSquareImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # create test image
        cls.img1 = np.arange(100, 10001, 100).reshape((10, 10))

        cls.img1[5:6, 7:8] = 1
        cls.img1[5:9, 1:3] = 1

        cls.img1[0:3, 5:8] = np.repeat(np.arange(2, 5), 3).reshape((3, 3))
        cls.img1[7:10, 5:8] = np.repeat(np.arange(2, 5), 3).reshape((3, 3))
        cls.img1[1:4, 1:4] = np.tile(np.arange(2, 5), 3).reshape((3, 3))

        cls.epsilon1 = 1.1

        # create test data from slow cpu approach (which was tested in test 1)
        dm = grqa.distance_matrix(cls.img1)
        cls.rp = grqa.recurrence_plot(dm, epsilon=cls.epsilon1)

        rp_sum1, ind1, len1, dirs1, dir_fac1 = grqa.diag_hs_patch_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum2, ind2, len2, dirs2, dir_fac2 = grqa.diag_hs_patch_sampling(
            cls.rp, rot=True, return_distribution=False
        )
        rp_sum3, ind3, len3, dirs3, dir_fac3 = grqa.diag_hs_line_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum4, ind4, len4, dirs4, dir_fac4 = grqa.diag_hs_line_sampling(
            cls.rp, rot=True, return_distribution=False
        )

        rp_sum5, ind5, len5, dirs5, dir_fac5 = grqa.vert_hs_patch_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum6, ind6, len6, dirs6, dir_fac6 = grqa.vert_hs_patch_sampling(
            cls.rp, rot=True, return_distribution=False
        )
        rp_sum7, ind7, len7, dirs7, dir_fac7 = grqa.vert_hs_line_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum8, ind8, len8, dirs8, dir_fac8 = grqa.vert_hs_line_sampling(
            cls.rp, rot=True, return_distribution=False
        )

        # store frequencies for testing
        cls.frequ_arr = np.zeros((8, cls.img1.shape[0] - 1))  # -1, bc minlen=2
        for i, len in enumerate([len1, len2, len3, len4, len5, len6, len7, len8]):
            frequency, lengths = np.histogram(len, bins=np.arange(2, np.max(len1) + 2))
            cls.frequ_arr[i, :] = frequency
        cls.lengths = lengths[:-1]

        # store RP sum for testing
        assert (
            rp_sum1
            == rp_sum2
            == rp_sum3
            == rp_sum4
            == rp_sum5
            == rp_sum6
            == rp_sum7
            == rp_sum8
        ), "RP sums do not match of the slow cpu approach do not match."
        cls.rp_sum = rp_sum1

        # store direction factors for testing
        cls.dir_fac_arr = np.array(
            [
                dir_fac1,
                dir_fac2,
                dir_fac3,
                dir_fac4,
                dir_fac5,
                dir_fac6,
                dir_fac7,
                dir_fac8,
            ]
        )

    def test_slow_cpu(self):

        rot_arg = [False, True] * 4
        for i, func in enumerate(
            [
                grqa.diag_hs_patch_sampling,
                grqa.diag_hs_patch_sampling,
                grqa.diag_hs_line_sampling,
                grqa.diag_hs_line_sampling,
                grqa.vert_hs_patch_sampling,
                grqa.vert_hs_patch_sampling,
                grqa.vert_hs_line_sampling,
                grqa.vert_hs_line_sampling,
            ]
        ):
            rp_sum1, lengths1, frequency1, dir_fac1 = func(self.rp, rot=rot_arg[i])
            np.testing.assert_array_equal(frequency1, self.frequ_arr[i, :])
            assert rp_sum1 == self.rp_sum, "RP sums do not match."
            assert dir_fac1 == self.dir_fac_arr[i], "Direction factors do not match."

    def test_fast_cpu(self):

        rot_arg = [False, True] * 4
        for i, func in enumerate(
            [
                grqa.diag_hs_patch_sampling_fast,
                grqa.diag_hs_patch_sampling_fast,
                grqa.diag_hs_line_sampling_fast,
                grqa.diag_hs_line_sampling_fast,
                grqa.vert_hs_patch_sampling_fast,
                grqa.vert_hs_patch_sampling_fast,
                grqa.vert_hs_line_sampling_fast,
                grqa.vert_hs_line_sampling_fast,
            ]
        ):
            rp_sum1, lengths1, frequency1, dir_fac1, rr1 = func(
                self.img1, rot=rot_arg[i], epsilon=self.epsilon1
            )
            # they won't be exactly the same due to e.g. only considering the upper half for the diagonal sampling
            np.testing.assert_allclose(frequency1[:3], self.frequ_arr[i, :3], rtol=0.2)
            np.testing.assert_array_equal(frequency1[3:], self.frequ_arr[i, 3:])
            assert rp_sum1 == self.rp_sum, "RP sums do not match."
            assert dir_fac1 == self.dir_fac_arr[i], "Direction factors do not match."

    def test_fast_cpu_v2(self):
        rp_sum1, lengths1, frequency_diag1, frequency_vert1, dir_fac_diag1, dir_fac_vert1, rr1 = grqa.hs_line_sampling_fast(
            self.img1, rot=True, epsilon=self.epsilon1)

        # they won't be exactly the same due to e.g. only considering the upper half for the diagonal sampling
        np.testing.assert_allclose(frequency_diag1[:3], self.frequ_arr[3, :3], rtol=0.2)
        np.testing.assert_array_equal(frequency_diag1[3:], self.frequ_arr[3, 3:])
        # for the vertical lines if will differ a bit more here due to only considering upper half
        np.testing.assert_allclose(frequency_vert1, self.frequ_arr[7, :], rtol=0.2, atol=50)
        assert rp_sum1 == self.rp_sum, "RP sums do not match."
        assert dir_fac_diag1 == self.dir_fac_arr[3], "Direction factors do not match."
        assert dir_fac_vert1 == self.dir_fac_arr[7], "Direction factors do not match."

    def test_gpu(self):

        if cuda.is_available():
            rp_sum1, lengths1, frequency_diag1, frequency_vert1, dir_fac_diag1, dir_fac_vert1, rr1 = grqa.hs_line_sampling_gpu(
                self.img1, rot=True, epsilon=self.epsilon1)

            # they won't be exactly the same due to e.g. only considering the upper half for the diagonal sampling
            np.testing.assert_allclose(frequency_diag1[:3], self.frequ_arr[3, :3], rtol=0.2)
            np.testing.assert_array_equal(frequency_diag1[3:], self.frequ_arr[3, 3:])
            # for the vertical lines if will differ a bit more here due to only considering upper half
            np.testing.assert_allclose(frequency_vert1, self.frequ_arr[7, :], rtol=0.2, atol=50)
            assert rp_sum1 == self.rp_sum, "RP sums do not match."
            assert dir_fac_diag1 == self.dir_fac_arr[3], "Direction factors do not match."
            assert dir_fac_vert1 == self.dir_fac_arr[7], "Direction factors do not match."
        else:
            print("No GPU available, test can't be run.")




class TestRectangularImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # create test image
        cls.img1 = np.arange(100, 10001, 100).reshape((10, 10))

        cls.img1[5:6, 7:8] = 1
        cls.img1[5:9, 1:3] = 1

        cls.img1[0:3, 5:8] = np.repeat(np.arange(2, 5), 3).reshape((3, 3))
        cls.img1[7:10, 5:8] = np.repeat(np.arange(2, 5), 3).reshape((3, 3))
        cls.img1[1:4, 1:4] = np.tile(np.arange(2, 5), 3).reshape((3, 3))

        cls.epsilon1 = 1.1

        # create test data from slow cpu approach (which was tested in test 1)
        dm = grqa.distance_matrix(cls.img1)
        cls.rp = grqa.recurrence_plot(dm, epsilon=cls.epsilon1)

        rp_sum1, ind1, len1, dirs1, dir_fac1 = grqa.diag_hs_patch_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum2, ind2, len2, dirs2, dir_fac2 = grqa.diag_hs_patch_sampling(
            cls.rp, rot=True, return_distribution=False
        )
        rp_sum3, ind3, len3, dirs3, dir_fac3 = grqa.diag_hs_line_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum4, ind4, len4, dirs4, dir_fac4 = grqa.diag_hs_line_sampling(
            cls.rp, rot=True, return_distribution=False
        )

        rp_sum5, ind5, len5, dirs5, dir_fac5 = grqa.vert_hs_patch_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum6, ind6, len6, dirs6, dir_fac6 = grqa.vert_hs_patch_sampling(
            cls.rp, rot=True, return_distribution=False
        )
        rp_sum7, ind7, len7, dirs7, dir_fac7 = grqa.vert_hs_line_sampling(
            cls.rp, return_distribution=False
        )
        rp_sum8, ind8, len8, dirs8, dir_fac8 = grqa.vert_hs_line_sampling(
            cls.rp, rot=True, return_distribution=False
        )

        # store frequencies for testing
        cls.frequ_arr = np.zeros((8, cls.img1.shape[0] - 1))  # -1, bc minlen=2
        for i, len in enumerate([len1, len2, len3, len4, len5, len6, len7, len8]):
            frequency, lengths = np.histogram(len, bins=np.arange(2, np.max(len1) + 2))
            cls.frequ_arr[i, :] = frequency
        cls.lengths = lengths[:-1]

        # store RP sum for testing
        assert (
            rp_sum1
            == rp_sum2
            == rp_sum3
            == rp_sum4
            == rp_sum5
            == rp_sum6
            == rp_sum7
            == rp_sum8
        ), "RP sums do not match of the slow cpu approach do not match."
        cls.rp_sum = rp_sum1

        # store direction factors for testing
        cls.dir_fac_arr = np.array(
            [
                dir_fac1,
                dir_fac2,
                dir_fac3,
                dir_fac4,
                dir_fac5,
                dir_fac6,
                dir_fac7,
                dir_fac8,
            ]
        )

        cls.img2 = np.arange(100, 15001, 100).reshape((15, 10))

        cls.img2[8:9, 7:8] = 1
        cls.img2[8:12, 1:3] = 1

        cls.img2[3:6, 5:8] = np.repeat(np.arange(2, 5), 3).reshape((3, 3))
        cls.img2[10:13, 5:8] = np.repeat(np.arange(2, 5), 3).reshape((3, 3))
        cls.img2[4:7, 1:4] = np.tile(np.arange(2, 5), 3).reshape((3, 3))

        cls.epsilon2 = 1.1

        # create test data from slow cpu approach (which was tested in test 1)
        dm2 = grqa.distance_matrix(cls.img2)
        cls.rp2 = grqa.recurrence_plot(dm2, epsilon=cls.epsilon2)

    def test_slow_cpu(self):

        rot_arg = [False, True] * 4
        for i, func in enumerate(
            [
                grqa.diag_hs_patch_sampling,
                grqa.diag_hs_patch_sampling,
                grqa.diag_hs_line_sampling,
                grqa.diag_hs_line_sampling,
                grqa.vert_hs_patch_sampling,
                grqa.vert_hs_patch_sampling,
                grqa.vert_hs_line_sampling,
                grqa.vert_hs_line_sampling,
            ]
        ):
            rp_sum2, lengths2, frequency2, dir_fac2 = func(self.rp2, rot=rot_arg[i])

            # correct for longer diagonal HS and rectangular shape
            if (func == grqa.diag_hs_line_sampling) and (rot_arg[i] == False):
                # for rectangular image 4 additional diagonal lines of length 3 will be detected
                # (9, 2, 9, 2) and (11, 7, 11, 5) in direction (1, 0, 1, 1) and vise versa
                frequency2[1] -= 4
                # at line length 10, 5 less lines will be detected, because they are now length 15 (up up)
                frequency2[8] += 5
            elif (func == grqa.diag_hs_line_sampling) and (rot_arg[i] == True):
                # check out utils.get_differing_indices to get the differences between the two images
                frequency2[0] -= 8
                frequency2[1] -= 16
                frequency2[8] -= 10
            elif func == grqa.diag_hs_patch_sampling:
                frequency2[3] -= 2 * np.sqrt(dir_fac2)

            np.testing.assert_array_equal(frequency2[:9], self.frequ_arr[i, :])
            assert rp_sum2 == self.rp_sum + 5 * 10, "RP sums do not match."
            assert dir_fac2 == self.dir_fac_arr[i], "Direction factors do not match."

            if func == grqa.diag_hs_line_sampling:
                assert (
                    frequency2[-1] != 0
                ), "Diagonal lines of length 15 should be detected."
            else:
                assert frequency2[-1] == 0, "Lines of length 15 should not be detected."

    def test_fast_cpu(self):

        rot_arg = [False, True] * 4
        for i, func in enumerate(
            [
                grqa.diag_hs_patch_sampling_fast,
                grqa.diag_hs_patch_sampling_fast,
                grqa.diag_hs_line_sampling_fast,
                grqa.diag_hs_line_sampling_fast,
                grqa.vert_hs_patch_sampling_fast,
                grqa.vert_hs_patch_sampling_fast,
                grqa.vert_hs_line_sampling_fast,
                grqa.vert_hs_line_sampling_fast,
            ]
        ):
            rp_sum2, lengths2, frequency2, dir_fac2, rr2 = func(
                self.img2, rot=rot_arg[i], epsilon=self.epsilon2
            )

            if (func == grqa.diag_hs_line_sampling_fast) and (rot_arg[i] == False):
                frequency2[1] -= 4
                frequency2[8] += 5
            elif (func == grqa.diag_hs_line_sampling_fast) and (rot_arg[i] == True):
                frequency2[0] -= 8
                frequency2[1] -= 16
                frequency2[8] -= 10
            elif func == grqa.diag_hs_patch_sampling_fast:
                frequency2[3] -= 2 * np.sqrt(dir_fac2)

            np.testing.assert_allclose(frequency2[:3], self.frequ_arr[i, :3], rtol=0.2)
            np.testing.assert_array_equal(frequency2[3:9], self.frequ_arr[i, 3:])
            assert rp_sum2 == self.rp_sum + 5 * 10, "RP sums do not match."
            assert dir_fac2 == self.dir_fac_arr[i], "Direction factors do not match."

            if func == grqa.diag_hs_line_sampling_fast:
                assert (
                    frequency2[-1] != 0
                ), "Diagonal lines of length 15 should be detected."
            else:
                assert frequency2[-1] == 0, "Lines of length 15 should not be detected."

    def test_fast_cpu_v2(self):
        rp_sum1, lengths1, frequency_diag1, frequency_vert1, dir_fac_diag1, dir_fac_vert1, rr1 = grqa.hs_line_sampling_fast(
            self.img2, rot=True, epsilon=self.epsilon1)

        # new structures for rectangular shape
        frequency_diag1[0] -= 8
        frequency_diag1[1] -= 16
        frequency_diag1[8] -= 10

        # diagonal lines will differ due to only sampling the upper half
        np.testing.assert_allclose(frequency_diag1[:3], self.frequ_arr[3, :3], rtol=0.2)
        np.testing.assert_array_equal(frequency_diag1[3:9], self.frequ_arr[3, 3:])
        # vertical lines will differ more due to only sampling the upper half
        np.testing.assert_allclose(frequency_vert1[:9], self.frequ_arr[7, :], rtol=0.5, atol=50)
        assert rp_sum1 == self.rp_sum + 5 * 10, "RP sums do not match." # additional pixels for diag HS
        assert dir_fac_diag1 == self.dir_fac_arr[3], "Direction factors do not match."
        assert dir_fac_vert1 == self.dir_fac_arr[7], "Direction factors do not match."

    def test_gpu(self):

        if cuda.is_available():
            rp_sum1, lengths1, frequency_diag1, frequency_vert1, dir_fac_diag1, dir_fac_vert1, rr1 = grqa.hs_line_sampling_fast(
                self.img2, rot=True, epsilon=self.epsilon1)

            # new structures for rectangular shape
            frequency_diag1[0] -= 8
            frequency_diag1[1] -= 16
            frequency_diag1[8] -= 10

            # diagonal lines will differ due to only sampling the upper half
            np.testing.assert_allclose(frequency_diag1[:3], self.frequ_arr[3, :3], rtol=0.2)
            np.testing.assert_array_equal(frequency_diag1[3:9], self.frequ_arr[3, 3:])
            # vertical lines will differ more due to only sampling the upper half
            np.testing.assert_allclose(frequency_vert1[:9], self.frequ_arr[7, :], rtol=0.5, atol=50)
            assert rp_sum1 == self.rp_sum + 5 * 10, "RP sums do not match."  # additional pixels for diag HS
            assert dir_fac_diag1 == self.dir_fac_arr[3], "Direction factors do not match."
            assert dir_fac_vert1 == self.dir_fac_arr[7], "Direction factors do not match."
        else:
            print("No GPU available, test can't be run.")


class TestMeasures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.det_list = []
        cls.l_hs_list = []
        cls.ent_list = []
        cls.lam_list = []
        cls.tt_list = []

    def apply_to_diagonal_measures(
        self,
        diag_line_lengths,
        diag_line_len,
        diag_line_frequ,
        diag_patch_sizes,
        diag_patch_s,
        diag_patch_frequ,
        diag_dir_fac_line,
        diag_dir_fac_patch,
        diag_rp_sum,
        det_expect,
        l_hs_expect,
        ent_expect,
        test_func=np.testing.assert_equal,
        test_func_args={},
    ):
        det_list = []
        l_hs_list = []
        ent_list = []

        det, l_hs = grqa.determinism_and_avg_diag_hs_size(
            diag_line_frequ, diag_line_len, diag_rp_sum, dir_fac=diag_dir_fac_line
        )
        det_list.append(det)
        l_hs_list.append(l_hs)
        test_func(det, det_expect, **test_func_args)
        test_func(l_hs, l_hs_expect, **test_func_args)

        det, l_hs = grqa.determinism_and_avg_diag_hs_size_patch(
            diag_patch_frequ, diag_patch_s, diag_rp_sum, dir_fac=diag_dir_fac_patch
        )
        det_list.append(det)
        l_hs_list.append(l_hs)
        test_func(det, det_expect, **test_func_args)
        test_func(l_hs, l_hs_expect, **test_func_args)

        det, l_hs = grqa.determinism_and_avg_diag_hs_size_from_lengths(
            diag_rp_sum, diag_line_lengths, dir_fac=diag_dir_fac_line
        )
        det_list.append(det)
        l_hs_list.append(l_hs)
        test_func(det, det_expect, **test_func_args)
        test_func(l_hs, l_hs_expect, **test_func_args)

        det, l_hs = grqa.determinism_and_avg_diag_hs_size_patch_from_lengths(
            diag_rp_sum, diag_patch_sizes, dir_fac=diag_dir_fac_patch
        )
        det_list.append(det)
        l_hs_list.append(l_hs)
        test_func(det, det_expect, **test_func_args)
        test_func(l_hs, l_hs_expect, **test_func_args)

        ent = grqa.entropy(diag_line_frequ)
        ent_list.append(ent)
        test_func(ent, ent_expect, **test_func_args)

        ent = grqa.entropy_from_lengths(diag_line_lengths)
        ent_list.append(ent)
        test_func(ent, ent_expect, **test_func_args)

        ent = grqa.entropy(diag_patch_frequ)
        ent_list.append(ent)
        test_func(ent, ent_expect, **test_func_args)

        ent = grqa.entropy_from_lengths(diag_patch_sizes)
        ent_list.append(ent)
        test_func(ent, ent_expect, **test_func_args)

        return np.array(det_list), np.array(l_hs_list), np.array(ent_list)

    def apply_to_vertical_measures(
        self,
        vert_line_lengths,
        vert_line_len,
        vert_line_frequ,
        vert_patch_sizes,
        vert_patch_s,
        vert_patch_frequ,
        vert_dir_fac_line,
        vert_dir_fac_patch,
        vert_rp_sum,
        lam_expect,
        tt_expect,
        test_func=np.testing.assert_equal,
        test_func_args={},
    ):
        lam_list = []
        tt_list = []

        lam, tt = grqa.laminarity_and_trapping_size(
            vert_line_frequ, vert_line_len, vert_rp_sum, dir_fac=vert_dir_fac_line
        )
        lam_list.append(lam)
        tt_list.append(tt)
        test_func(lam, lam_expect, **test_func_args)
        test_func(tt, tt_expect, **test_func_args)

        lam, tt = grqa.laminarity_and_trapping_size_patch(
            vert_patch_frequ, vert_patch_s, vert_rp_sum, dir_fac=vert_dir_fac_patch
        )
        lam_list.append(lam)
        tt_list.append(tt)
        test_func(lam, lam_expect, **test_func_args)
        test_func(tt, tt_expect, **test_func_args)

        lam, tt = grqa.laminarity_and_trapping_size_from_lengths(
            vert_rp_sum, vert_line_lengths, dir_fac=vert_dir_fac_line
        )
        lam_list.append(lam)
        tt_list.append(tt)
        test_func(lam, lam_expect, **test_func_args)
        test_func(tt, tt_expect, **test_func_args)

        lam, tt = grqa.laminarity_and_trapping_size_patch_from_lengths(
            vert_rp_sum, vert_patch_sizes, dir_fac=vert_dir_fac_patch
        )
        lam_list.append(lam)
        tt_list.append(tt)
        test_func(lam, lam_expect, **test_func_args)
        test_func(tt, tt_expect, **test_func_args)

        return np.array(lam_list), np.array(tt_list)

    def test_predictable_image(self):
        # (all rp pixels that are 1 belong to a structure)

        # one 10 x 10 diagonal HS that gets sampled in 1 direction (up up)
        diag_line_lengths = np.ones(10) * 10
        diag_line_len, diag_line_frequ = np.array([10]), np.array([10])
        diag_patch_sizes = np.array([10])
        diag_patch_s, diag_patch_frequ = np.array([10]), np.array([1])
        diag_dir_fac = 1
        diag_rp_sum = 10 * 10

        det_expect = 1
        l_hs_expect = 10
        ent_expect = 0

        self.apply_to_diagonal_measures(
            diag_line_lengths,
            diag_line_len,
            diag_line_frequ,
            diag_patch_sizes,
            diag_patch_s,
            diag_patch_frequ,
            diag_dir_fac,
            diag_dir_fac,
            diag_rp_sum,
            det_expect,
            l_hs_expect,
            ent_expect,
        )

        # one 10 x 10 vertical HS that gets sampled in 1 direction (up up)
        vert_line_lengths = np.ones(10) * 10
        vert_line_len, vert_line_frequ = np.array([10]), np.array([10])
        vert_patch_sizes = np.array([10])
        vert_patch_s, vert_patch_frequ = np.array([10]), np.array([1])
        vert_dir_fac = 1
        vert_rp_sum = 10 * 10

        lam_expect = 1
        tt_expect = 10

        self.apply_to_vertical_measures(
            vert_line_lengths,
            vert_line_len,
            vert_line_frequ,
            vert_patch_sizes,
            vert_patch_s,
            vert_patch_frequ,
            vert_dir_fac,
            vert_dir_fac,
            vert_rp_sum,
            lam_expect,
            tt_expect,
        )

    def test_unpredictable_image(self):
        # (no pixel that is 1 in the rp is part of a structure)

        # 100 random pixels that are not connected to each other
        diag_line_lengths = np.array([])
        diag_line_len, diag_line_frequ = np.array([]), np.array([])
        diag_patch_sizes = np.array([])
        diag_patch_s, diag_patch_frequ = np.array([]), np.array([])
        diag_dir_fac = 1
        diag_rp_sum = 10 * 10

        det_expect = 0
        l_hs_expect = 0
        ent_expect = np.nan

        self.apply_to_diagonal_measures(
            diag_line_lengths,
            diag_line_len,
            diag_line_frequ,
            diag_patch_sizes,
            diag_patch_s,
            diag_patch_frequ,
            diag_dir_fac,
            diag_dir_fac,
            diag_rp_sum,
            det_expect,
            l_hs_expect,
            ent_expect,
        )

        # 100 random pixels that are not connected to each other
        vert_line_lengths = np.array([])
        vert_line_len, vert_line_frequ = np.array([]), np.array([])
        vert_patch_sizes = np.array([])
        vert_patch_s, vert_patch_frequ = np.array([]), np.array([])
        vert_dir_fac = np.array([])
        vert_rp_sum = np.array([])

        lam_expect = 0
        tt_expect = 0

        self.apply_to_vertical_measures(
            vert_line_lengths,
            vert_line_len,
            vert_line_frequ,
            vert_patch_sizes,
            vert_patch_s,
            vert_patch_frequ,
            vert_dir_fac,
            vert_dir_fac,
            vert_rp_sum,
            lam_expect,
            tt_expect,
        )

    def test_random_vs_periodic(self):

        ### test white noise image (< 0.2)

        size = 20
        whitenoise = np.random.uniform(0, 1, (size, size))

        dm = grqa.distance_matrix(whitenoise)
        epsilon = grqa.recurrence_threshold(dm, 0.2)
        rp = grqa.recurrence_plot(dm, epsilon)

        diag_rp_sum, _, diag_line_lengths, _, diag_dir_fac_line = (
            grqa.diag_hs_line_sampling(rp, return_distribution=False)
        )
        _, diag_line_len, diag_line_frequ, _ = grqa.diag_hs_line_sampling(rp)
        _, _, diag_patch_sizes, _, diag_dir_fac_patch = grqa.diag_hs_patch_sampling(
            rp, return_distribution=False
        )
        _, diag_patch_s, diag_patch_frequ, _ = grqa.diag_hs_patch_sampling(rp)

        det_expect = 0.35
        l_hs_expect = 3
        ent_expect = 0.6

        det_list_ran, l_hs_list_ran, ent_list_ran = self.apply_to_diagonal_measures(
            diag_line_lengths,
            diag_line_len,
            diag_line_frequ,
            diag_patch_sizes,
            diag_patch_s,
            diag_patch_frequ,
            diag_dir_fac_line,
            diag_dir_fac_patch,
            diag_rp_sum,
            det_expect,
            l_hs_expect,
            ent_expect,
            test_func=np.testing.assert_array_less,
        )

        vert_rp_sum, _, vert_line_lengths, _, vert_dir_fac_line = (
            grqa.vert_hs_line_sampling(rp, return_distribution=False)
        )
        _, vert_line_len, vert_line_frequ, _ = grqa.vert_hs_line_sampling(rp)
        _, _, vert_patch_sizes, _, vert_dir_fac_patch = grqa.vert_hs_patch_sampling(
            rp, return_distribution=False
        )
        _, vert_patch_s, vert_patch_frequ, _ = grqa.vert_hs_patch_sampling(rp)

        lam_expect = 0.15
        tt_expect = 2.2

        lam_list_ran, tt_list_ran = self.apply_to_vertical_measures(
            vert_line_lengths,
            vert_line_len,
            vert_line_frequ,
            vert_patch_sizes,
            vert_patch_s,
            vert_patch_frequ,
            vert_dir_fac_line,
            vert_dir_fac_patch,
            vert_rp_sum,
            lam_expect,
            tt_expect,
            test_func=np.testing.assert_allclose,
            test_func_args={"atol": 0.2, "rtol": 0.1},
        )

        ### test regular pattern (> 0.8)

        size = 25
        x = np.linspace(0, 4 * np.pi, size)
        y = np.linspace(0, 0 * np.pi, size)
        X, Y = np.meshgrid(x, y)
        periodic = np.sin(X) + np.cos(Y)

        dm = grqa.distance_matrix(periodic)
        epsilon = grqa.recurrence_threshold(dm, 0.2)
        rp = grqa.recurrence_plot(dm, epsilon)

        diag_rp_sum, _, diag_line_lengths, _, diag_dir_fac_line = (
            grqa.diag_hs_line_sampling(rp, return_distribution=False)
        )
        _, diag_line_len, diag_line_frequ, _ = grqa.diag_hs_line_sampling(rp)
        _, _, diag_patch_sizes, _, diag_dir_fac_patch = grqa.diag_hs_patch_sampling(
            rp, return_distribution=False
        )
        _, diag_patch_s, diag_patch_frequ, _ = grqa.diag_hs_patch_sampling(rp)

        det_expect = 0.9
        l_hs_expect = 3.5
        ent_expect = 1.2

        det_list_reg, l_hs_list_reg, ent_list_reg = self.apply_to_diagonal_measures(
            diag_line_lengths,
            diag_line_len,
            diag_line_frequ,
            diag_patch_sizes,
            diag_patch_s,
            diag_patch_frequ,
            diag_dir_fac_line,
            diag_dir_fac_patch,
            diag_rp_sum,
            det_expect,
            l_hs_expect,
            ent_expect,
            test_func=np.testing.assert_allclose,
            test_func_args={"atol": 0.2, "rtol": 0.5},
        )

        vert_rp_sum, _, vert_line_lengths, _, vert_dir_fac_line = (
            grqa.vert_hs_line_sampling(rp, return_distribution=False)
        )
        _, vert_line_len, vert_line_frequ, _ = grqa.vert_hs_line_sampling(rp)
        _, _, vert_patch_sizes, _, vert_dir_fac_patch = grqa.vert_hs_patch_sampling(
            rp, return_distribution=False
        )
        _, vert_patch_s, vert_patch_frequ, _ = grqa.vert_hs_patch_sampling(rp)

        lam_expect = 0.9
        tt_expect = 5

        lam_list_reg, tt_list_reg = self.apply_to_vertical_measures(
            vert_line_lengths,
            vert_line_len,
            vert_line_frequ,
            vert_patch_sizes,
            vert_patch_s,
            vert_patch_frequ,
            vert_dir_fac_line,
            diag_dir_fac_patch,
            vert_rp_sum,
            lam_expect,
            tt_expect,
            test_func=np.testing.assert_allclose,
            test_func_args={"atol": 0.2, "rtol": 0.5},
        )

        ### combine

        np.testing.assert_array_less(det_list_ran, det_list_reg)
        np.testing.assert_array_less(l_hs_list_ran, l_hs_list_reg)
        np.testing.assert_array_less(ent_list_ran, ent_list_reg)
        np.testing.assert_array_less(lam_list_ran, lam_list_reg)
        np.testing.assert_array_less(tt_list_ran, tt_list_reg)


if __name__ == "__main__":
    unittest.main()
