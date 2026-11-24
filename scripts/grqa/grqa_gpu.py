import itertools
import time

import numpy as np
from numba import cuda, int32, types
from scipy.spatial.distance import pdist

from scripts import utils


# a useless decorator for when cuda is not available
def useless_decorator(func):
    print("WARNING: No GPU available.")
    return func


# cuda decorators for when cuda is available
if cuda.is_available():
    kernel_decorator_line_lengths = cuda.jit(
        "float32[::, ::], int32[::, ::], int32[::, ::], int64[::], int64[::], int32, int32, float32, int64[::]"
    )
    kernel_decorator_mass_distribution = cuda.jit(
        "float32[::, ::], int32[::, ::], int64[::], int64[::], int64[::], int64[::], int64[::], int64[::], int32, int32, int32, float32"
    )
    kernel_decorator_mass_distribution_rand = cuda.jit(
        "float32[::, ::], int32[::, ::], int64[::], int64[::], int64[::], int64[::], int64[::], int64[::], int64[::], int32, int32, int32, int32, float32"
    )
    kernel_decorator_mass_distribution_large = cuda.jit(
        "float32[::, ::], int32[::, ::], int32[::], int64[::], int64[::], int64[::], int64[::], int64[::], int64[::], int32, int32, int32, int32, int32, int32, int32, float32"
    )
else:
    kernel_decorator_line_lengths = useless_decorator
    kernel_decorator_mass_distribution = useless_decorator
    kernel_decorator_mass_distribution_rand = useless_decorator
    kernel_decorator_mass_distribution_large = useless_decorator


# %% line length distribution


@kernel_decorator_line_lengths
def get_line_lengths_cuda(
    img, grid, patterns, frequency_diag, frequency_vert, ys, xs, epsilon_mod, rp_sum
):
    """
    Get the frequency of diagonal and vertical lines. Each thread is processing one hyper-voxel.
    """

    img = cuda.const.array_like(img)
    grid = cuda.const.array_like(grid)
    patterns = cuda.const.array_like(patterns)

    i, j = cuda.grid(2)

    y1 = grid[0, i]
    x1 = grid[1, i]
    y2 = grid[0, j]
    x2 = grid[1, j]

    # check if thread indices are within img bounds
    if i >= ys * xs or j >= ys * xs:
        return

    # check if pixel is on the upper half
    if y2 < y1:
        return

    # in the "y-line" of (y1, x1), if (y2 == y1), the x2 value has to be
    # larger than x1 to avoid going through the same recurrences twice
    if (y1 == y2) and (x1 >= x2):
        return

    # check if the pixel is 1
    # (otherwise it can't be on a line)
    if abs(img[y1, x1] - img[y2, x2]) > epsilon_mod:
        return

    # add to the rp sum
    cuda.atomic.add(rp_sum, 0, 1)

    # go through all directions and find line lengths
    for p_nr, rot_p in enumerate(patterns):
        k1 = rot_p[0]
        k2 = rot_p[1]
        k3 = rot_p[2]
        k4 = rot_p[3]

        # check if the pixel is not on the edge or of the pixel on the opposite direction is not 0
        # if pixel is e.g. on the right edge and bottom left pixel is checked, it would still
        # only be l = 1 which is < lmin
        if (
            (0 <= y1 - k1 < ys)
            and (0 <= x1 - k2 < xs)
            and (0 <= y2 - k3 < ys)
            and (0 <= x2 - k4 < xs)
            and (abs(img[y1 - k1, x1 - k2] - img[y2 - k3, x2 - k4]) <= epsilon_mod)
        ):
            continue

        # go along diagonal line to find the length
        l = 0
        d = 1

        while d == 1:
            l += 1

            # if l == ys: # max_dim
            #    d = 0
            if (
                (0 <= y1 + k1 * l < ys)
                and (0 <= x1 + k2 * l < xs)
                and (0 <= y2 + k3 * l < ys)
                and (0 <= x2 + k4 * l < xs)
                and (
                    abs(img[y1 + k1 * l, x1 + k2 * l] - img[y2 + k3 * l, x2 + k4 * l])
                    <= epsilon_mod
                )
            ):
                d = 1
            else:
                d = 0

        if (l > 1) and (k1 == 0) and (k2 == 0):
            cuda.atomic.add(frequency_vert, int32(l), 1)
        elif l > 1:
            cuda.atomic.add(frequency_diag, int32(l), 1)


def hs_line_sampling_gpu(
    img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False, nthreads=(16, 16)
):

    ### image as c-contiguous array

    if not img.data.c_contiguous:
        img = np.ascontiguousarray(img)

    ### define grid

    ys, xs = img.shape
    yy, xx = np.meshgrid(range(ys), range(xs))
    grid = np.vstack([yy.ravel(), xx.ravel()]).astype(np.int32)

    ### define rotation patterns

    # the rotation patterns are the factors for k1, k2, k3 and k4
    # rotation only in 45° steps
    # (k1, k2) != (0, 0) for diagonal lines, (k1, k2) == (0, 0) for vertical lines
    # (k3, k4) != (0, 0)
    if not rot:
        rot_pattern = [(1, 0), (1, 1)]
    else:
        rot_pattern = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ]

    # all combinations of k values
    patterns = (
        np.array(list(itertools.product(rot_pattern, rot_pattern)))
        .flatten()
        .reshape((len(rot_pattern) ** 2, 4))
    )
    patterns = np.concatenate(
        [patterns, np.array([(0, 0, i, j) for i, j in rot_pattern])]
    )

    # filtering out mirroring directions
    try:
        mirr_idx = np.array(
            [
                np.where(np.sum(patterns == -patterns[i, :], axis=1) == 4)[0][0]
                for i in range(patterns.shape[0])
            ]
        )
        idx_unique = np.unique(
            np.sort(np.vstack([np.arange(patterns.shape[0]), mirr_idx]), axis=0), axis=1
        )[0]
    except IndexError:
        idx_unique = np.arange(patterns.shape[0])
    patterns = patterns[idx_unique, :]
    patterns = patterns.astype(np.int32)

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    ### initialize length and frequency arrays and recurrence plot sum

    max_dim = max([ys, xs])
    lengths = np.arange(max_dim + 1)
    frequency_diag = np.zeros((max_dim + 1), dtype=np.int64)
    frequency_vert = np.zeros((max_dim + 1), dtype=np.int64)
    rp_sum = np.zeros(1, dtype=np.int64)

    ### get line length distributions

    if cuda.is_available():
        time1 = time.time()
        print("Copying data to GPU...")
        d_img = cuda.to_device(img.astype(np.float32))
        d_grid = cuda.to_device(grid)
        d_patterns = cuda.to_device(patterns)
        d_frequency_diag = cuda.to_device(frequency_diag)
        d_frequency_vert = cuda.to_device(frequency_vert)
        d_rp_sum = cuda.to_device(rp_sum)

        nblocksy = ys * xs // nthreads[0] + 1
        nblocksx = xs * ys // nthreads[1] + 1

        nblocks = (nblocksy, nblocksx)

        time2 = time.time()

        get_line_lengths_cuda[nblocks, nthreads](
            d_img,
            d_grid,
            d_patterns,
            d_frequency_diag,
            d_frequency_vert,
            ys,
            xs,
            epsilon_mod,
            d_rp_sum,
        )
        cuda.synchronize()

        time3 = time.time()
        print(f"Copying data back to host... (run time: {time3 - time2})")
        frequency_diag = d_frequency_diag.copy_to_host()
        frequency_vert = d_frequency_vert.copy_to_host()
        rp_sum = d_rp_sum.copy_to_host()

        print(
            f"copy back time: {time.time() - time3}, total time: {time.time() - time1}"
        )
    else:
        print("No GPU available.")
        return 0, [], [], [], 0, 0, 0

    ### diagonal frequency: account for only evaluating the upper half and half of the directions

    # double the frequency bc only upper half
    # BEFORE adding the main diagonal
    frequency_diag = frequency_diag * 2

    # adjust the rp sum (upper half)
    rp_sum = rp_sum * 2

    # add the main diagonal
    if rot:
        # add the main HS sampled by diagonal lines
        frequency_diag[list(range(minlen, np.min([xs, ys]) + 1))] += 2
        frequency_diag[list(range(minlen, np.min([xs, ys])))] += 2
        frequency_diag[np.min([ys, xs])] += 2 * np.abs(ys - xs)

        # add the main HS sampled by straight lines
        frequency_diag[ys] += xs
        frequency_diag[xs] += ys

        # adjust recurrence plot sum
        rp_sum += ys * xs

        # adjust frequency (mirroring directions)
        frequency_diag *= 2

        # set direction factor for rotation
        dir_fac_diag = 64
    else:
        # add the main HS sampled by diagonal lines
        frequency_diag[list(range(minlen, np.min([xs, ys]) + 1))] += 1
        frequency_diag[list(range(minlen, np.min([xs, ys])))] += 1
        frequency_diag[np.min([ys, xs])] += np.abs(ys - xs)

        # add the main HS sampled by straight lines
        frequency_diag[ys] += xs

        rp_sum += ys * xs

        dir_fac_diag = 4

    ### vertical frequency: account for only evaluating the upper half and half of the directions

    # double the frequency bc only upper half
    # BEFORE adding the main diagonal
    frequency_vert = frequency_vert * 2

    # add the main diagonal
    if rot:
        dir_fac_vert = 8
        frequency_vert *= 2  # (mirroring dirs)
    else:
        dir_fac_vert = 2

    # only return the lengths >= minlen
    lengths = lengths[minlen:]
    frequency_diag = frequency_diag[minlen:]
    frequency_vert = frequency_vert[minlen:]

    if verbose:
        try:
            print(f"{np.sum(frequency_diag)} diagonal lines of lenghts:")
            # print(np.where(frequency_diag != 0)[0] + minlen)
            print(
                np.min(np.where(frequency_diag != 0)[0] + minlen),
                "...",
                np.max(np.where(frequency_diag != 0)[0] + minlen),
            )
            print("were detected.")
        except:
            print("No diagonal lines detected.")

        try:
            print(f"{np.sum(frequency_vert)} vertical lines of lenghts:")
            # print(np.where(frequency_vert != 0)[0] + minlen)
            print(
                np.min(np.where(frequency_vert != 0)[0] + minlen),
                "...",
                np.max(np.where(frequency_vert != 0)[0] + minlen),
            )
            print("were detected.")
        except:
            print("No vertical lines detected.")

        print(f"The recurrence plot sum is: {rp_sum}.")
        print("The recurrence rate is:", rp_sum / (ys**2 * xs**2))

    return (
        rp_sum,
        lengths,
        frequency_diag,
        frequency_vert,
        dir_fac_diag,
        dir_fac_vert,
        rp_sum / (ys**2 * xs**2),
    )


# %% mass distribution


def mass_distribution_gpu(
    img, epsilon=None, RR=0.2, max_box_size=None, nthreads=(16, 16)
):
    """
    Go through each hyper-voxel and add the weight of each hyper-voxel to the according box.
    No storing of intermediate results, no significance assessment.
    """

    ### image as c-contiguous array

    if not img.data.c_contiguous:
        img = np.ascontiguousarray(img)

    ### define grid

    ys, xs = img.shape
    yy, xx = np.meshgrid(range(ys), range(xs))
    grid = np.vstack([yy.ravel(), xx.ravel()]).astype(np.int32)

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    ### set maximum box size

    if not max_box_size:
        max_box_size = int(0.1 * max(ys, xs))

    ### defining the cuda function within the function to allow different sizing of lacal arrays

    # size of the local arrays
    local_size = max_box_size + 1

    # maximum value for overflow
    max_value = 10**18

    @kernel_decorator_mass_distribution
    def get_mass_distribution_cuda(
        img,
        grid,
        mass_distribution,
        mass_distribution_overflow,
        mass_distribution_z2,
        mass_distribution_z2_overflow,
        mass_distribution_reverse_z2,
        mass_distribution_reverse_z2_overflow,
        ys,
        xs,
        max_box_size,
        epsilon_mod,
    ):
        """
        Get the mass distribution. Each thread is processing one hyper-voxel.
        """

        img = cuda.const.array_like(img)
        grid = cuda.const.array_like(grid)

        i, j = cuda.grid(2)

        # create mass distribution for each pixel
        mass_distribution_px = cuda.local.array(local_size, dtype=types.int64)
        mass_distribution_reverse_px = cuda.local.array(local_size, dtype=types.int64)

        # zero initialization of the local arrays
        # for idx in range(local_size):
        #    mass_distribution_px[idx] = 0
        #    mass_distribution_reverse_px[idx] = 0

        # check if thread indices are within img bounds
        if i >= ys * xs or j >= ys * xs:
            return

        y1 = grid[0, i]
        x1 = grid[1, i]
        y2 = grid[0, j]
        x2 = grid[1, j]

        new_max_box_size = max_box_size
        # calculate maximum box size that needs to be calculated for this pixel
        for new_max_box_size in range(max_box_size, 0, -1):
            if (
                y1 % new_max_box_size
                + x1 % new_max_box_size
                + y2 % new_max_box_size
                + x2 % new_max_box_size
                == 0
            ):
                break
        new_max_box_size = min(
            new_max_box_size, min(ys - y1, xs - x1, ys - y2, xs - x2)
        )

        for w1 in range(1, new_max_box_size + 1):
            for w2 in range(1, new_max_box_size + 1):
                for w3 in range(1, new_max_box_size + 1):
                    for w4 in range(1, new_max_box_size + 1):

                        w = max(w1, w2, w3, w4)

                        # not necessary because already checked above
                        # if ((y1 + w > ys) or
                        #        (x1 + w > xs) or
                        #        (y2 + w > ys) or
                        #        (x2 + w > xs)):
                        #    continue

                        weight = int(
                            abs(
                                img[y1 + w1 - 1, x1 + w2 - 1]
                                - img[y2 + w3 - 1, x2 + w4 - 1]
                            )
                            <= epsilon_mod
                        )

                        for idx in range(w, new_max_box_size + 1):
                            mass_distribution_px[idx] += weight
                            mass_distribution_reverse_px[idx] += 1 - weight

        # cuda.syncthreads()
        for idx in range(1, max_box_size + 1):
            if y1 % idx + x1 % idx + y2 % idx + x2 % idx == 0:
                # cuda.atomic.add(mass_distribution, idx, mass_distribution_px[idx])
                # cuda.atomic.add(mass_distribution_z2, idx, mass_distribution_px[idx] ** 2)
                # cuda.atomic.add(mass_distribution_reverse_z2, idx, mass_distribution_reverse_px[idx]**2)

                if max_value - mass_distribution[idx] < mass_distribution_px[idx]:
                    cuda.atomic.add(mass_distribution_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution, idx, -(max_value - mass_distribution_px[idx])
                    )
                else:
                    cuda.atomic.add(mass_distribution, idx, mass_distribution_px[idx])

                if (
                    max_value - mass_distribution_z2[idx]
                    < mass_distribution_px[idx] ** 2
                ):
                    cuda.atomic.add(mass_distribution_z2_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution_z2,
                        idx,
                        -(max_value - mass_distribution_px[idx] ** 2),
                    )
                else:
                    cuda.atomic.add(
                        mass_distribution_z2, idx, mass_distribution_px[idx] ** 2
                    )

                if (
                    max_value - mass_distribution_reverse_z2[idx]
                    < mass_distribution_reverse_px[idx] ** 2
                ):
                    cuda.atomic.add(mass_distribution_reverse_z2_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution_reverse_z2,
                        idx,
                        -(max_value - mass_distribution_reverse_px[idx] ** 2),
                    )
                else:
                    cuda.atomic.add(
                        mass_distribution_reverse_z2,
                        idx,
                        mass_distribution_reverse_px[idx] ** 2,
                    )

    ### set up mass distribution counters

    mass_distribution = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_overflow = np.zeros((max_box_size + 1), dtype=np.int64)
    # mass_distribution_reverse = np.zeros((max_box_size+1), dtype=np.int64)
    mass_distribution_z2 = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_z2_overflow = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse_z2 = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse_z2_overflow = np.zeros((max_box_size + 1), dtype=np.int64)

    if cuda.is_available():

        ### copy arrays to device

        time1 = time.time()
        print("Copying data to GPU...")
        d_img = cuda.to_device(img.astype(np.float32))
        d_grid = cuda.to_device(grid)
        d_mass_distribution = cuda.to_device(mass_distribution)
        d_mass_distribution_overflow = cuda.to_device(mass_distribution_overflow)
        d_mass_distribution_z2 = cuda.to_device(mass_distribution_z2)
        d_mass_distribution_z2_overflow = cuda.to_device(mass_distribution_z2_overflow)
        d_mass_distribution_reverse_z2 = cuda.to_device(mass_distribution_reverse_z2)
        d_mass_distribution_reverse_z2_overflow = cuda.to_device(
            mass_distribution_reverse_z2_overflow
        )

        nblocksy = ys * xs // nthreads[0] + 1
        nblocksx = xs * ys // nthreads[1] + 1

        nblocks = (nblocksy, nblocksx)

        time2 = time.time()
        print(
            f"Calculating the mass of all boxes up to size {max_box_size}... (copy time: {time2 - time1})"
        )
        get_mass_distribution_cuda[nblocks, nthreads](
            d_img,
            d_grid,
            d_mass_distribution,
            d_mass_distribution_overflow,
            d_mass_distribution_z2,
            d_mass_distribution_z2_overflow,
            d_mass_distribution_reverse_z2,
            d_mass_distribution_reverse_z2_overflow,
            ys,
            xs,
            max_box_size,
            epsilon_mod,
        )
        cuda.synchronize()

        time3 = time.time()
        print(f"Copying data back to host... (run time: {time3 - time2})")
        mass_distribution = d_mass_distribution.copy_to_host()
        mass_distribution_overflow = d_mass_distribution_overflow.copy_to_host()
        mass_distribution_z2 = d_mass_distribution_z2.copy_to_host()
        mass_distribution_z2_overflow = d_mass_distribution_z2_overflow.copy_to_host()
        mass_distribution_reverse_z2 = d_mass_distribution_reverse_z2.copy_to_host()
        mass_distribution_reverse_z2_overflow = (
            d_mass_distribution_reverse_z2_overflow.copy_to_host()
        )
        print(
            f"copy back time: {time.time() - time3}, total time: {time.time() - time1}"
        )

        mass_distribution = mass_distribution.astype(
            object
        ) + max_value * mass_distribution_overflow.astype(object)
        mass_distribution_z2 = mass_distribution_z2.astype(
            object
        ) + max_value * mass_distribution_z2_overflow.astype(object)
        mass_distribution_reverse_z2 = mass_distribution_reverse_z2.astype(
            object
        ) + max_value * mass_distribution_reverse_z2_overflow.astype(object)

        # for overlapping boxes
        # nr_boxes = np.hstack([np.zeros(1), ((xs - np.arange(1, max_box_size+1)) * (ys - np.arange(1, max_box_size+1))) ** 2])

        # for non overlapping boxes
        nr_boxes = np.hstack(
            [
                np.zeros(1),
                (
                    (ys // np.arange(1, max_box_size + 1))
                    * (xs // np.arange(1, max_box_size + 1))
                )
                ** 2,
            ]
        )
        mass_distribution_reverse = (
            nr_boxes * np.arange(max_box_size + 1) ** 4 - mass_distribution
        ).astype(object)
    else:
        nr_boxes = np.hstack(
            [
                np.zeros(1),
                (
                    (ys // np.arange(1, max_box_size + 1))
                    * (xs // np.arange(1, max_box_size + 1))
                )
                ** 2,
            ]
        )
        mass_distribution_reverse = np.zeros_like(mass_distribution)

    return (
        mass_distribution,
        mass_distribution_reverse,
        mass_distribution_z2,
        mass_distribution_reverse_z2,
        nr_boxes,
    )


def mass_distribution_gpu_rand(
    img, epsilon=None, RR=0.2, max_box_size=None, nthreads=(16, 16), sample_size=100
):
    """
    Go through each hyper-voxel and add the weight of each pixel to the according hyper-voxel.
    Draw number of random boxes from a fixed grid (n = sample_size)
    """

    ### image as c-contiguous array

    if not img.data.c_contiguous:
        img = np.ascontiguousarray(img)

    ### define grid

    ys, xs = img.shape
    # yy, xx = np.meshgrid(range(ys), range(xs))
    # grid = np.vstack([yy.ravel(), xx.ravel()]).astype(np.int32)

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    ### set maximum box size

    if not max_box_size:
        max_box_size = int(0.1 * max(ys, xs))

    ### create grid

    yy, xx = np.meshgrid(range(ys), range(xs))
    grid = np.vstack([yy.ravel(), xx.ravel()]).astype(np.int32)

    assert grid.shape[1] >= sample_size, "Choose a smaller sample size."

    grid_rand = grid[
        :, np.random.choice(range(grid.shape[1]), sample_size, replace=False)
    ]
    # grid_rand = grid
    print(grid_rand)

    ### defining the cuda function within the function to allow different sizing of lacal arrays

    # size of the local arrays
    local_size = max_box_size + 1

    # maximum value for overflow
    max_value = 10**18

    @kernel_decorator_mass_distribution_rand
    def get_mass_distribution_cuda_rand(
        img,
        grid,
        mass_distribution,
        mass_distribution_overflow,
        mass_distribution_z2,
        mass_distribution_z2_overflow,
        mass_distribution_reverse_z2,
        mass_distribution_reverse_z2_overflow,
        nr_boxes,
        ys,
        xs,
        grid_lim,
        max_box_size,
        epsilon_mod,
    ):

        img = cuda.const.array_like(img)
        grid = cuda.const.array_like(grid)

        i, j = cuda.grid(2)

        # create mass distribution for each pixel
        mass_distribution_px = cuda.local.array(local_size, dtype=types.int64)
        mass_distribution_reverse_px = cuda.local.array(local_size, dtype=types.int64)

        # zero initialization of the local arrays
        # for idx in range(local_size):
        #    mass_distribution_px[idx] = 0
        #    mass_distribution_reverse_px[idx] = 0

        # check if thread indices are within img bounds
        if i >= grid_lim or j >= grid_lim:
            return

        y1 = grid[0, i]
        x1 = grid[1, i]
        y2 = grid[0, j]
        x2 = grid[1, j]

        new_max_box_size = max_box_size
        # calculate maximum box size that needs to be calculated for this pixel
        for new_max_box_size in range(max_box_size, 0, -1):
            if (
                y1 % new_max_box_size
                + x1 % new_max_box_size
                + y2 % new_max_box_size
                + x2 % new_max_box_size
                == 0
            ):
                break
        new_max_box_size = min(
            new_max_box_size, min(ys - y1, xs - x1, ys - y2, xs - x2)
        )

        for w1 in range(1, new_max_box_size + 1):
            for w2 in range(1, new_max_box_size + 1):
                for w3 in range(1, new_max_box_size + 1):
                    for w4 in range(1, new_max_box_size + 1):

                        w = max(w1, w2, w3, w4)

                        # not necessary because already checked above
                        # if ((y1 + w > ys) or
                        #        (x1 + w > xs) or
                        #        (y2 + w > ys) or
                        #        (x2 + w > xs)):
                        #    continue

                        weight = int(
                            abs(
                                img[y1 + w1 - 1, x1 + w2 - 1]
                                - img[y2 + w3 - 1, x2 + w4 - 1]
                            )
                            <= epsilon_mod
                        )

                        for idx in range(w, new_max_box_size + 1):
                            mass_distribution_px[idx] += weight
                            mass_distribution_reverse_px[idx] += 1 - weight

        # cuda.syncthreads()
        for idx in range(1, new_max_box_size + 1):
            if y1 % idx + x1 % idx + y2 % idx + x2 % idx == 0:
                # cuda.atomic.add(mass_distribution, idx, mass_distribution_px[idx])
                # cuda.atomic.add(mass_distribution_z2, idx, mass_distribution_px[idx] ** 2)
                # cuda.atomic.add(mass_distribution_reverse_z2, idx, mass_distribution_reverse_px[idx]**2)

                if max_value - mass_distribution[idx] < mass_distribution_px[idx]:
                    cuda.atomic.add(mass_distribution_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution, idx, -(max_value - mass_distribution_px[idx])
                    )
                else:
                    cuda.atomic.add(mass_distribution, idx, mass_distribution_px[idx])

                if (
                    max_value - mass_distribution_z2[idx]
                    < mass_distribution_px[idx] ** 2
                ):
                    cuda.atomic.add(mass_distribution_z2_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution_z2,
                        idx,
                        -(max_value - mass_distribution_px[idx] ** 2),
                    )
                else:
                    cuda.atomic.add(
                        mass_distribution_z2, idx, mass_distribution_px[idx] ** 2
                    )

                if (
                    max_value - mass_distribution_reverse_z2[idx]
                    < mass_distribution_reverse_px[idx] ** 2
                ):
                    cuda.atomic.add(mass_distribution_reverse_z2_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution_reverse_z2,
                        idx,
                        -(max_value - mass_distribution_reverse_px[idx] ** 2),
                    )
                else:
                    cuda.atomic.add(
                        mass_distribution_reverse_z2,
                        idx,
                        mass_distribution_reverse_px[idx] ** 2,
                    )

                cuda.atomic.add(nr_boxes, idx, 1)

    ### set up mass distribution counters

    mass_distribution = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_overflow = np.zeros((max_box_size + 1), dtype=np.int64)
    # mass_distribution_reverse = np.zeros((max_box_size+1), dtype=np.int64)
    mass_distribution_z2 = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_z2_overflow = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse_z2 = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse_z2_overflow = np.zeros((max_box_size + 1), dtype=np.int64)
    nr_boxes = np.zeros((max_box_size + 1), dtype=np.int64)

    if cuda.is_available():

        ### copy arrays to device

        time1 = time.time()
        print("Copying data to GPU...")
        d_img = cuda.to_device(img.astype(np.float32))
        d_grid = cuda.to_device(grid_rand)
        d_mass_distribution = cuda.to_device(mass_distribution)
        d_mass_distribution_overflow = cuda.to_device(mass_distribution_overflow)
        d_mass_distribution_z2 = cuda.to_device(mass_distribution_z2)
        d_mass_distribution_z2_overflow = cuda.to_device(mass_distribution_z2_overflow)
        d_mass_distribution_reverse_z2 = cuda.to_device(mass_distribution_reverse_z2)
        d_mass_distribution_reverse_z2_overflow = cuda.to_device(
            mass_distribution_reverse_z2_overflow
        )
        d_nr_boxes = cuda.to_device(nr_boxes)

        grid_lim = grid_rand.shape[1]
        nblocksy = grid_lim // nthreads[0] + 1
        nblocksx = grid_lim // nthreads[1] + 1

        nblocks = (nblocksy, nblocksx)

        time2 = time.time()
        print(
            f"Calculating the mass of all boxes up to size {max_box_size}... (copy time: {time2 - time1})"
        )
        get_mass_distribution_cuda_rand[nblocks, nthreads](
            d_img,
            d_grid,
            d_mass_distribution,
            d_mass_distribution_overflow,
            d_mass_distribution_z2,
            d_mass_distribution_z2_overflow,
            d_mass_distribution_reverse_z2,
            d_mass_distribution_reverse_z2_overflow,
            d_nr_boxes,
            ys,
            xs,
            grid_lim,
            max_box_size,
            epsilon_mod,
        )
        cuda.synchronize()

        time3 = time.time()
        print(f"Copying data back to host... (run time: {time3 - time2})")
        mass_distribution = d_mass_distribution.copy_to_host()
        mass_distribution_overflow = d_mass_distribution_overflow.copy_to_host()
        mass_distribution_z2 = d_mass_distribution_z2.copy_to_host()
        mass_distribution_z2_overflow = d_mass_distribution_z2_overflow.copy_to_host()
        mass_distribution_reverse_z2 = d_mass_distribution_reverse_z2.copy_to_host()
        mass_distribution_reverse_z2_overflow = (
            d_mass_distribution_reverse_z2_overflow.copy_to_host()
        )
        nr_boxes = d_nr_boxes.copy_to_host()
        print(
            f"copy back time: {time.time() - time3}, total time: {time.time() - time1}"
        )

        mass_distribution = mass_distribution.astype(
            object
        ) + max_value * mass_distribution_overflow.astype(object)
        mass_distribution_z2 = mass_distribution_z2.astype(
            object
        ) + max_value * mass_distribution_z2_overflow.astype(object)
        mass_distribution_reverse_z2 = mass_distribution_reverse_z2.astype(
            object
        ) + max_value * mass_distribution_reverse_z2_overflow.astype(object)
        mass_distribution_reverse = (
            nr_boxes * np.arange(max_box_size + 1) ** 4 - mass_distribution
        ).astype(object)
    else:
        nr_boxes = np.hstack(
            [
                np.zeros(1),
                (
                    (ys // np.arange(1, max_box_size + 1))
                    * (xs // np.arange(1, max_box_size + 1))
                )
                ** 2,
            ]
        )
        mass_distribution_reverse = np.zeros_like(mass_distribution)

    return (
        mass_distribution,
        mass_distribution_reverse,
        mass_distribution_z2,
        mass_distribution_reverse_z2,
        nr_boxes,
    )


def mass_distribution_gpu_large(
    img,
    epsilon=None,
    RR=0.2,
    max_box_size=None,
    nthreads=(16, 16),
    box_size_store=50,
    max_box_size_large=100,
):
    """
    Go through each hyper-voxel and add the weight of each hyper-voxel to the according box.
    Save intermediate results of a certain box size box_size_store.
    Every box size until max_box_size gets evaluated with a step size of 1.
    The rest, until max_box_size_large, gets evaluated until max_box_size_large.
    """

    ### image as c-contiguous array

    if not img.data.c_contiguous:
        img = np.ascontiguousarray(img)

    ### define grid

    ys, xs = img.shape
    yy, xx = np.meshgrid(range(ys), range(xs))
    grid = np.vstack([yy.ravel(), xx.ravel()]).astype(np.int32)

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    ### set maximum box size

    if not max_box_size:
        max_box_size = int(0.1 * max(ys, xs))

    ### defining the cuda function within the function to allow different sizing of lacal arrays

    # size of the local arrays
    local_size = max_box_size_large + 1

    # size of shared array
    large_grid_size_y = ys // box_size_store
    large_grid_size_x = xs // box_size_store
    device_array_size = large_grid_size_y**2 * large_grid_size_x**2

    # maximum value for overflow
    pot = 18
    max_value = 10**pot
    sqrt_pot = int((pot - 1) / 2)

    # check
    if max_box_size_large**8 >= max_value:
        print(
            "WARNING: Results for large box sizes might be unreliable,\n"
            + "check the mass distributions for signs of undefined behavior"
        )

    @cuda.jit
    def square_large_number(a):
        """
        Square numbers whose square does not fit in int64.
        """

        # split the number at sqrt_pot
        a_upper = int(a / 10 ** (sqrt_pot + 1))
        a_lower = int(a - a_upper * 10 ** (sqrt_pot + 1))

        # middle term
        term_upper = int(a_upper * a_lower / 10 ** (sqrt_pot + 1))
        term_lower = int(a_upper * a_lower - term_upper * 10 ** (sqrt_pot + 1))

        c = 0
        if max_value - a_lower**2 <= (2 * term_lower) * 10 ** (sqrt_pot + 1):
            c = 1
            result_lower = a_lower**2 - (
                max_value - (2 * term_lower) * 10 ** (sqrt_pot + 1)
            )
        else:
            result_lower = a_lower**2 + (2 * term_lower) * 10 ** (sqrt_pot + 1)
        result_upper = a_upper**2 + 2 * term_upper + c

        return result_upper, result_lower

    @cuda.jit
    def add_values_to_global_arrays(
        local_size,
        y1,
        x1,
        y2,
        x2,
        mass_distribution_px,
        mass_distribution_reverse_px,
        mass_distribution,
        mass_distribution_overflow,
        mass_distribution_z2,
        mass_distribution_z2_overflow,
        mass_distribution_reverse_z2,
        mass_distribution_reverse_z2_overflow,
    ):
        """
        Add results from local arrays to global arrays by considering overflow.
        """

        # add the weights to the global mass distribution arrays
        for idx in range(1, local_size):
            if y1 % idx + x1 % idx + y2 % idx + x2 % idx == 0:

                if max_value - mass_distribution[idx] < mass_distribution_px[idx]:
                    cuda.atomic.add(mass_distribution_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution, idx, -(max_value - mass_distribution_px[idx])
                    )
                else:
                    cuda.atomic.add(mass_distribution, idx, mass_distribution_px[idx])

                if mass_distribution_px[idx] >= 10 ** (sqrt_pot + 1):
                    result_upper, result_lower = square_large_number(
                        mass_distribution_px[idx]
                    )
                    cuda.atomic.add(mass_distribution_z2_overflow, idx, result_upper)
                    if max_value - mass_distribution_z2[idx] < result_lower:
                        cuda.atomic.add(mass_distribution_z2_overflow, idx, 1)
                        cuda.atomic.add(
                            mass_distribution_z2,
                            idx,
                            -(max_value - result_lower),
                        )
                    else:
                        cuda.atomic.add(mass_distribution_z2, idx, result_lower)
                elif (
                    max_value - mass_distribution_z2[idx]
                    < mass_distribution_px[idx] ** 2
                ):
                    cuda.atomic.add(mass_distribution_z2_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution_z2,
                        idx,
                        -(max_value - mass_distribution_px[idx] ** 2),
                    )
                else:
                    cuda.atomic.add(
                        mass_distribution_z2, idx, mass_distribution_px[idx] ** 2
                    )

                if mass_distribution_reverse_px[idx] >= 10 ** (sqrt_pot + 1):
                    result_upper, result_lower = square_large_number(
                        mass_distribution_reverse_px[idx]
                    )
                    cuda.atomic.add(
                        mass_distribution_reverse_z2_overflow, idx, result_upper
                    )
                    if max_value - mass_distribution_reverse_z2[idx] < result_lower:
                        cuda.atomic.add(mass_distribution_reverse_z2_overflow, idx, 1)
                        cuda.atomic.add(
                            mass_distribution_reverse_z2,
                            idx,
                            -(max_value - result_lower),
                        )
                    else:
                        cuda.atomic.add(mass_distribution_reverse_z2, idx, result_lower)
                elif (
                    max_value - mass_distribution_reverse_z2[idx]
                    < mass_distribution_reverse_px[idx] ** 2
                ):
                    cuda.atomic.add(mass_distribution_reverse_z2_overflow, idx, 1)
                    cuda.atomic.add(
                        mass_distribution_reverse_z2,
                        idx,
                        -(max_value - mass_distribution_reverse_px[idx] ** 2),
                    )
                else:
                    cuda.atomic.add(
                        mass_distribution_reverse_z2,
                        idx,
                        mass_distribution_reverse_px[idx] ** 2,
                    )

    @cuda.jit("int32[::], int64")
    def clear_large_grid(large_grid, device_array_size):

        i, j = cuda.grid(2)
        xs_grid, ys_grid = cuda.gridsize(2)

        if j * xs_grid + i >= device_array_size:
            return

        idx = j * xs_grid + i
        large_grid[idx] = 0

    @kernel_decorator_mass_distribution_large
    def get_mass_distribution_cuda_large_step_1(
        img,
        grid,
        large_grid,
        mass_distribution,
        mass_distribution_overflow,
        mass_distribution_z2,
        mass_distribution_z2_overflow,
        mass_distribution_reverse_z2,
        mass_distribution_reverse_z2_overflow,
        ys,
        xs,
        max_box_size,
        box_size_store,
        max_box_size_large,
        large_grid_size_y,
        large_grid_size_x,
        epsilon_mod,
    ):
        """
        First kernel that calculates all box sizes with step size 1 until max_box_size and stores
        intermediate results for box size store in a global array.
        """

        # set image and grid to constant arrays
        img = cuda.const.array_like(img)
        grid = cuda.const.array_like(grid)

        # zero initialize large grid
        # for idx in range(device_array_size):
        #     large_grid[idx] = 0

        i, j = cuda.grid(2)

        # create mass distribution for each pixel
        mass_distribution_px = cuda.local.array(local_size, dtype=types.int64)
        mass_distribution_reverse_px = cuda.local.array(local_size, dtype=types.int64)

        # zero initialization of the local arrays
        # for idx in range(local_size):
        #    mass_distribution_px[idx] = 0
        #    mass_distribution_reverse_px[idx] = 0

        # check if thread indices are within img bounds
        if i >= ys * xs or j >= ys * xs:
            return

        y1 = grid[0, i]
        x1 = grid[1, i]
        y2 = grid[0, j]
        x2 = grid[1, j]

        new_max_box_size = max_box_size
        # calculate maximum box size that needs to be calculated for this pixel
        for new_max_box_size in range(max_box_size, 0, -1):
            if (
                y1 % new_max_box_size
                + x1 % new_max_box_size
                + y2 % new_max_box_size
                + x2 % new_max_box_size
                == 0
            ):
                break
        new_max_box_size = min(
            new_max_box_size, min(ys - y1, xs - x1, ys - y2, xs - x2)
        )

        # add weight of the individual pixels to the according boxes
        for w1 in range(1, new_max_box_size + 1):
            for w2 in range(1, new_max_box_size + 1):
                for w3 in range(1, new_max_box_size + 1):
                    for w4 in range(1, new_max_box_size + 1):

                        w = max(w1, w2, w3, w4)

                        weight = int(
                            abs(
                                img[y1 + w1 - 1, x1 + w2 - 1]
                                - img[y2 + w3 - 1, x2 + w4 - 1]
                            )
                            <= epsilon_mod
                        )

                        for idx in range(w, new_max_box_size + 1):
                            mass_distribution_px[idx] += weight
                            mass_distribution_reverse_px[idx] += 1 - weight

        # store the weight of the box with box_size_store in the large grid
        if (
            (
                y1 % box_size_store
                + x1 % box_size_store
                + y2 % box_size_store
                + x2 % box_size_store
                == 0
            )
            and (y1 // box_size_store < large_grid_size_y)
            and (x1 // box_size_store < large_grid_size_x)
            and (y2 // box_size_store < large_grid_size_y)
            and (x2 // box_size_store < large_grid_size_x)
        ):

            index = (
                (y1 // box_size_store)
                + (x1 // box_size_store) * large_grid_size_y
                + (y2 // box_size_store) * large_grid_size_x * large_grid_size_y
                + (x2 // box_size_store) * large_grid_size_x * large_grid_size_y**2
            )

            if index < device_array_size:
                cuda.atomic.add(large_grid, index, mass_distribution_px[box_size_store])

        # set the mass for the pixels in the large grid back to zero, they will be added in step 2
        for n in range(max_box_size_large // box_size_store):
            mass_distribution_px[n * box_size_store] = 0
            mass_distribution_reverse_px[n * box_size_store] = 0

        add_values_to_global_arrays(
            local_size,
            y1,
            x1,
            y2,
            x2,
            mass_distribution_px,
            mass_distribution_reverse_px,
            mass_distribution,
            mass_distribution_overflow,
            mass_distribution_z2,
            mass_distribution_z2_overflow,
            mass_distribution_reverse_z2,
            mass_distribution_reverse_z2_overflow,
        )

    @kernel_decorator_mass_distribution_large
    def get_mass_distribution_cuda_large_step_2(
        img,
        grid,
        large_grid,
        mass_distribution,
        mass_distribution_overflow,
        mass_distribution_z2,
        mass_distribution_z2_overflow,
        mass_distribution_reverse_z2,
        mass_distribution_reverse_z2_overflow,
        ys,
        xs,
        max_box_size,
        box_size_store,
        max_box_size_large,
        large_grid_size_y,
        large_grid_size_x,
        epsilon_mod,
    ):
        """
        Second kernel that calculates box sizes up until max_box_size_large by using the intermediate results
        and therefore a step size of box_size_store.
        """

        i, j = cuda.grid(2)

        # create mass distribution for each pixel
        mass_distribution_px = cuda.local.array(local_size, dtype=types.int64)
        mass_distribution_reverse_px = cuda.local.array(local_size, dtype=types.int64)

        # zero initialization of the local arrays
        for idx in range(local_size):
            mass_distribution_px[idx] = 0
            mass_distribution_reverse_px[idx] = 0

        # check if thread indices are within img bounds
        if i >= ys * xs or j >= ys * xs:
            return

        y1 = grid[0, i]
        x1 = grid[1, i]
        y2 = grid[0, j]
        x2 = grid[1, j]

        # only work with the pixels that are on the large grid
        if not (
            (
                y1 % box_size_store
                + x1 % box_size_store
                + y2 % box_size_store
                + x2 % box_size_store
                == 0
            )
            and (y1 // box_size_store < large_grid_size_y)
            and (x1 // box_size_store < large_grid_size_x)
            and (y2 // box_size_store < large_grid_size_y)
            and (x2 // box_size_store < large_grid_size_x)
        ):
            return

        # get the new possible maximum box size
        new_max_box_size = min(
            max_box_size_large // box_size_store,
            large_grid_size_y - y1 // box_size_store,
            large_grid_size_x - x1 // box_size_store,
            large_grid_size_y - y2 // box_size_store,
            large_grid_size_x - x2 // box_size_store,
        )

        # get the values for the boxes on the large grid up until max_box_size_large
        for w1 in range(1, new_max_box_size + 1):
            for w2 in range(1, new_max_box_size + 1):
                for w3 in range(1, new_max_box_size + 1):
                    for w4 in range(1, new_max_box_size + 1):

                        w = max(w1, w2, w3, w4)

                        index = (
                            (y1 // box_size_store + w1 - 1)
                            + (x1 // box_size_store + w2 - 1) * large_grid_size_y
                            + (y2 // box_size_store + w3 - 1)
                            * large_grid_size_x
                            * large_grid_size_y
                            + (x2 // box_size_store + w4 - 1)
                            * large_grid_size_x
                            * large_grid_size_y**2
                        )
                        if index < device_array_size:
                            weight = large_grid[index]
                        else:
                            weight = 0

                        for idx in range(w, new_max_box_size + 1):
                            mass_distribution_px[idx * box_size_store] += weight
                            mass_distribution_reverse_px[idx * box_size_store] += (
                                box_size_store**4 - weight
                            )

        add_values_to_global_arrays(
            local_size,
            y1,
            x1,
            y2,
            x2,
            mass_distribution_px,
            mass_distribution_reverse_px,
            mass_distribution,
            mass_distribution_overflow,
            mass_distribution_z2,
            mass_distribution_z2_overflow,
            mass_distribution_reverse_z2,
            mass_distribution_reverse_z2_overflow,
        )

    ### set up mass distribution counters

    mass_distribution = np.zeros((max_box_size_large + 1), dtype=np.int64)
    mass_distribution_overflow = np.zeros((max_box_size_large + 1), dtype=np.int64)
    # mass_distribution_reverse = np.zeros((max_box_size_large+1), dtype=np.int64)
    mass_distribution_z2 = np.zeros((max_box_size_large + 1), dtype=np.int64)
    mass_distribution_z2_overflow = np.zeros((max_box_size_large + 1), dtype=np.int64)
    mass_distribution_reverse_z2 = np.zeros((max_box_size_large + 1), dtype=np.int64)
    mass_distribution_reverse_z2_overflow = np.zeros(
        (max_box_size_large + 1), dtype=np.int64
    )

    if cuda.is_available():

        ### copy arrays to device

        time1 = time.time()
        print("Copying data to GPU...")
        d_img = cuda.to_device(img.astype(np.float32))
        d_grid = cuda.to_device(grid)
        d_mass_distribution = cuda.to_device(mass_distribution)
        d_mass_distribution_overflow = cuda.to_device(mass_distribution_overflow)
        d_mass_distribution_z2 = cuda.to_device(mass_distribution_z2)
        d_mass_distribution_z2_overflow = cuda.to_device(mass_distribution_z2_overflow)
        d_mass_distribution_reverse_z2 = cuda.to_device(mass_distribution_reverse_z2)
        d_mass_distribution_reverse_z2_overflow = cuda.to_device(
            mass_distribution_reverse_z2_overflow
        )

        large_grid = cuda.device_array(
            (
                large_grid_size_y
                * large_grid_size_x
                * large_grid_size_y
                * large_grid_size_x
            ),
            dtype=np.int32,
        )

        ### get block size

        nblocksy = ys * xs // nthreads[0] + 1
        nblocksx = xs * ys // nthreads[1] + 1

        nblocks = (nblocksy, nblocksx)

        time2 = time.time()
        print(
            f"Calculating the mass of all boxes up to size {max_box_size_large}... (copy time: {time2 - time1})"
        )

        ### clear large grid

        print("Clean kernel...")
        clear_large_grid[nblocks, nthreads](large_grid, device_array_size)

        cuda.synchronize()

        ### start kernel 1

        get_mass_distribution_cuda_large_step_1[nblocks, nthreads](
            d_img,
            d_grid,
            large_grid,
            d_mass_distribution,
            d_mass_distribution_overflow,
            d_mass_distribution_z2,
            d_mass_distribution_z2_overflow,
            d_mass_distribution_reverse_z2,
            d_mass_distribution_reverse_z2_overflow,
            ys,
            xs,
            max_box_size,
            box_size_store,
            max_box_size_large,
            large_grid_size_y,
            large_grid_size_x,
            epsilon_mod,
        )
        cuda.synchronize()

        # large_grid = large_grid.copy_to_host()
        # print(large_grid)
        # print(np.unique(large_grid))

        ### start kernel 2

        get_mass_distribution_cuda_large_step_2[nblocks, nthreads](
            d_img,
            d_grid,
            large_grid,
            d_mass_distribution,
            d_mass_distribution_overflow,
            d_mass_distribution_z2,
            d_mass_distribution_z2_overflow,
            d_mass_distribution_reverse_z2,
            d_mass_distribution_reverse_z2_overflow,
            ys,
            xs,
            max_box_size,
            box_size_store,
            max_box_size_large,
            large_grid_size_y,
            large_grid_size_x,
            epsilon_mod,
        )
        cuda.synchronize()

        time3 = time.time()
        print(f"Copying data back to host... (run time: {time3 - time2})")

        ### copy data back to host

        mass_distribution = d_mass_distribution.copy_to_host()
        mass_distribution_overflow = d_mass_distribution_overflow.copy_to_host()
        mass_distribution_z2 = d_mass_distribution_z2.copy_to_host()
        mass_distribution_z2_overflow = d_mass_distribution_z2_overflow.copy_to_host()
        mass_distribution_reverse_z2 = d_mass_distribution_reverse_z2.copy_to_host()
        mass_distribution_reverse_z2_overflow = (
            d_mass_distribution_reverse_z2_overflow.copy_to_host()
        )
        print(
            f"copy back time: {time.time() - time3}, total time: {time.time() - time1}"
        )

        ### reset device

        # device = cuda.get_current_device()
        # device.reset()

        ### postprocess results

        mass_distribution = mass_distribution.astype(
            object
        ) + max_value * mass_distribution_overflow.astype(object)
        mass_distribution_z2 = mass_distribution_z2.astype(
            object
        ) + max_value * mass_distribution_z2_overflow.astype(object)
        mass_distribution_reverse_z2 = mass_distribution_reverse_z2.astype(
            object
        ) + max_value * mass_distribution_reverse_z2_overflow.astype(object)

        # for overlapping boxes
        # nr_boxes = np.hstack([np.zeros(1), ((xs - np.arange(1, max_box_size+1)) * (ys - np.arange(1, max_box_size+1))) ** 2])

        # for non overlapping boxes
        nr_boxes = np.hstack(
            [
                np.zeros(1),
                (
                    (ys // np.arange(1, max_box_size + 1))
                    * (xs // np.arange(1, max_box_size + 1))
                )
                ** 2,
                np.zeros(max_box_size_large - max_box_size),
            ]
        )
        nr_boxes[box_size_store : max_box_size_large + 1 : box_size_store] = (
            (ys // np.arange(box_size_store, max_box_size_large + 1, box_size_store))
            * (xs // np.arange(box_size_store, max_box_size_large + 1, box_size_store))
        ) ** 2
        mass_distribution_reverse = (
            nr_boxes * np.arange(max_box_size_large + 1) ** 4 - mass_distribution
        ).astype(object)
    else:
        nr_boxes = np.hstack(
            [
                np.zeros(1),
                (
                    (ys // np.arange(1, max_box_size_large + 1))
                    * (xs // np.arange(1, max_box_size_large + 1))
                )
                ** 2,
            ]
        )
        mass_distribution_reverse = np.zeros_like(mass_distribution)

    return (
        mass_distribution,
        mass_distribution_reverse,
        mass_distribution_z2,
        mass_distribution_reverse_z2,
        nr_boxes,
    )
