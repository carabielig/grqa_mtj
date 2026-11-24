import itertools

import numpy as np
from scipy.spatial.distance import pdist, squareform
from skimage.util import view_as_blocks
from tqdm import tqdm, trange

from scripts import utils

# %% patch sampling


def diag_hs_patch_sampling_fast(
    img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False
):
    """
    Get diagonal patch sizes (patch sampling approach).
    """

    ys, xs = img.shape
    img = np.ascontiguousarray(img)

    # the rotation patterns are the factors for k1 and k2
    if not rot:
        rot_pattern = [(1, 1)]
    else:
        rot_pattern = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    lengths = np.arange(np.max([ys, xs]) + 1)
    frequency = np.zeros(np.max([ys, xs]) + 1)

    i = 0

    if not epsilon:
        print("Estimating recurrence threshold...")
        est_px = np.min((ys, xs, 50))
        epsilon = np.quantile(
            pdist(
                np.c_[img[:est_px, :est_px].flatten(), img[:est_px, :est_px].flatten()],
                metric="euclidean",
            ),
            RR,
        )
        print("Epsilon:", epsilon)

    print(f"Detecting diagonal HS in {len(rot_pattern)**2} direction(s)...")
    for rot_p1 in rot_pattern:
        k1 = rot_p1[0]
        k2 = rot_p1[1]

        # for each direction the direction of the iteration through the coordinates
        # has to be changed, so that the HS get detected from the correct side
        if k1 < 0:
            yr1 = range(ys - 1, -1, k1)
        else:
            yr1 = range(ys)

        if k2 < 0:
            xr1 = range(xs - 1, -1, k2)
        else:
            xr1 = range(xs)

        for rot_p2 in rot_pattern:
            k3 = rot_p2[0]
            k4 = rot_p2[1]

            if k4 < 0:
                xr2 = range(xs - 1, -1, k4)
            else:
                xr2 = range(xs)

            print("Calculating Recurrence Plot...")
            assert epsilon >= 0, "Epsilon must be larger than 0."

            rp = squareform(
                pdist(np.c_[img.flatten(), img.flatten()], metric="euclidean")
            )
            rp = rp.reshape((img.shape[0], img.shape[1], img.shape[0], img.shape[1]))

            rp[rp <= epsilon] = 0
            rp[rp > epsilon] = 1
            rp = np.logical_not(rp).astype(int)

            rp_sum = np.sum(rp)

            # go through all coordinates (only upper half)
            # same conditions apply here as for the other methods
            # since it is just a selection of points
            for y1 in tqdm(yr1):
                for x1 in xr1:

                    if k3 < 0:
                        yr2 = range(ys - 1, y1 - 1, k3)
                    else:
                        yr2 = range(y1, ys)

                    for y2 in yr2:
                        for x2 in xr2:

                            # in the "y-line" of (y1, x1), if (y2 == y1), the x2 value has to be
                            # larger than x1 to avoid going through the same recurrences twice
                            if (y1 == y2) and (x1 >= x2):
                                continue

                            # check if the pixel is 1
                            # (otherwise it can't be on a HS)
                            if not rp[y1, x1, y2, x2] == 1:
                                continue

                            i += 1

                            # go along diagonal to find the length
                            l = -1
                            d = 1

                            while d == 1:
                                l += 1

                                if not (
                                    (0 <= y1 + k1 * l < ys)
                                    and (0 <= x1 + k2 * l < xs)
                                    and (0 <= y2 + k3 * l < ys)
                                    and (0 <= x2 + k4 * l < xs)
                                ):
                                    d = 0
                                else:
                                    d = rp[
                                        y1 + k1 * l,
                                        x1 + k2 * l,
                                        y2 + k3 * l,
                                        x2 + k4 * l,
                                    ]

                            # check y and x direction first to see if l is the actual length
                            # (because when HS intersect, then the diagonal can go through multiple HS)
                            y_sum = 0  # check y direction
                            for y_hs in range(l):
                                y_sum += rp[y1 + k1 * y_hs, x1, y2 + k3 * y_hs, x2]

                            x_sum = 0  # check x direction
                            for x_hs in range(l):
                                x_sum += rp[y1, x1 + k2 * x_hs, y2, x2 + k4 * x_hs]

                            l = int(min(y_sum, x_sum))

                            # check if really the whole HS is 1
                            hs_check = True
                            for k1_hs in range(l):
                                for k2_hs in range(l):

                                    if (
                                        rp[
                                            y1 + k1 * k1_hs,
                                            x1 + k2 * k2_hs,
                                            y2 + k3 * k1_hs,
                                            x2 + k4 * k2_hs,
                                        ]
                                        == 1
                                    ):
                                        continue
                                    else:
                                        hs_check = False

                            # if the HS check was passed, remove it from the rp and add the detected HS
                            if hs_check and (l >= minlen):
                                frequency[l] += 1

                                # remove whole HS from the rp to detect new HS, that
                                # were overlapping with this one
                                # (at this point the definition if diagonal HS is not true anymore)
                                for k1_hs in range(l):
                                    for k2_hs in range(l):
                                        rp[
                                            y1 + k1 * k1_hs,
                                            x1 + k2 * k2_hs,
                                            y2 + k3 * k1_hs,
                                            x2 + k4 * k2_hs,
                                        ] = 0

    print("Other RP sum:", i)

    # double the frequency bc only upper half
    # BEFORE adding the main diagonal
    frequency = frequency * 2

    # split the main diagonal HS
    lm = minlen + 1
    hs_len = []

    minimum = np.min((ys, xs))
    maximum = np.max((ys, xs))
    s = (maximum // minimum) * [minimum]

    while lm > minlen:
        hs_len.extend(s)
        ys_loc = maximum - (maximum // minimum) * minimum
        xs_loc = minimum

        minimum = np.max([np.min((ys_loc, xs_loc)), 1])
        maximum = np.max((ys_loc, xs_loc))
        s = (maximum // minimum) * [minimum]
        lm = np.min(s)

    if rot:
        # add main diagonal HS 4 times
        # use np.add because otherwise 4 will only be added once
        # for duplicate indices
        np.add.at(frequency, hs_len, 4)

        dir_fac = 16
    else:
        # add main diagonal HS 1 time
        np.add.at(frequency, hs_len, 1)

        dir_fac = 1

    # only return the lengths >= minlen
    lengths = lengths[minlen:]
    frequency = frequency[minlen:]

    if verbose:
        print(f"{np.sum(frequency)} diagonal patches of sizes:")
        print(np.where(frequency != 0)[0] + minlen)
        print("were detected.")
        print("The recurrence rate is:", rp_sum / (ys**2 * xs**2))

    return rp_sum, lengths, frequency, dir_fac, rp_sum / (ys**2 * xs**2)


def vert_hs_patch_sampling_fast(
    img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False
):
    """
    Get vertical patch sizes (patch sampling approach).
    """

    img = np.ascontiguousarray(img)

    # the rotation patterns are the factors for k1 and k2
    if not rot:
        rot_pattern = [(1, 1)]
    else:
        rot_pattern = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    ys, xs = img.shape

    lengths = np.arange(np.max([ys, xs]) + 1)
    frequency = np.zeros(np.max([ys, xs]) + 1)

    i = 0

    print(f"Detecting vertical HS in {len(rot_pattern)} direction(s)...")
    for rot_p in tqdm(rot_pattern):

        if not epsilon:
            print("Estimating recurrence threshold...")
            est_px = np.min((ys, xs, 50))
            epsilon = np.quantile(
                pdist(
                    np.c_[
                        img[:est_px, :est_px].flatten(), img[:est_px, :est_px].flatten()
                    ],
                    metric="euclidean",
                ),
                RR,
            )
        print("Epsilon:", epsilon)

        print("Calculating Recurrence Plot...")
        assert epsilon >= 0, "Epsilon must be larger than 0."

        rp = squareform(pdist(np.c_[img.flatten(), img.flatten()], metric="euclidean"))
        rp = rp.reshape((img.shape[0], img.shape[1], img.shape[0], img.shape[1]))

        rp[rp <= epsilon] = 0
        rp[rp > epsilon] = 1
        rp = np.logical_not(rp).astype(int)

        rp_sum = np.sum(rp)

        k1 = rot_p[0]
        k2 = rot_p[1]

        if k2 < 0:
            xr = range(rp.shape[1] - 1, -1, k2)
        else:
            xr = range(rp.shape[1])

        # go through all coordinates (only upper half)
        # same conditions apply here as for the other methods
        # since it is just a selection of points
        for y1 in range(ys):
            for x1 in range(xs):

                if k1 < 0:
                    yr = range(ys - 1, -1, k1)  ##### range(ys - 1, y1 - 1, k1)
                else:
                    yr = range(ys)  ##### range(y1, ys)

                for y2 in yr:
                    for x2 in xr:

                        # in the "y-line" of (y1, x1), if (y2 == y1), the x2 value has to be
                        # larger than x1 to avoid going through the same recurrences twice
                        ##### if (y1 == y2) and (x1 >= x2):
                        #####     continue

                        # check if the pixel is 1
                        # (otherwise it can't be on a HS)
                        if not rp[y1, x1, y2, x2] == 1:
                            continue

                        # go along diagonal to find the length
                        l = -1
                        d = 1

                        while d == 1:
                            l += 1

                            if not (
                                (0 <= y2 + k1 * l < ys) and (0 <= x2 + k2 * l < xs)
                            ):
                                d = 0
                            else:
                                d = rp[y1, x1, y2 + k1 * l, x2 + k2 * l]

                        # check y and x direction first to see if l is the actual length
                        # (because when HS intersect, then the diagonal can go through multiple HS)
                        y_sum = 0  # check y direction
                        for y_hs in range(l):
                            y_sum += rp[y1, x1, y2 + k1 * y_hs, x2]

                        x_sum = 0  # check x direction
                        for x_hs in range(l):
                            x_sum += rp[y1, x1, y2, x2 + k2 * x_hs]

                        l = int(min(y_sum, x_sum))

                        # the detected HS should be removed if it is >= 2 to also be able to detect maybe
                        # larger HS that intersect with the small one and would otherwise be lost
                        # check if really the whole HS is 1
                        hs_check = True
                        for k1_hs in range(l):
                            for k2_hs in range(l):

                                if rp[y1, x1, y2 + k1 * k1_hs, x2 + k2 * k2_hs] == 1:
                                    continue
                                else:
                                    hs_check = False

                        # if the HS check was passed, add the detected HS
                        if hs_check and (l >= minlen):

                            frequency[l] += 1

                            # remove whole HS from the rp to detect new HS, that
                            # were overlapping with this one
                            # (at this point the definition if diagonal HS is not true anymore)
                            for k1_hs in range(l):
                                for k2_hs in range(l):
                                    rp[y1, x1, y2 + k1 * k1_hs, x2 + k2 * k2_hs] = 0

    # double the frequency bc only upper half
    ##### frequency = frequency * 2
    # there is no main HS that needs to be added here, because there is none in the vertical direction
    # also, vertical HS can't cross the center

    # only return the lengths >= minlen
    lengths = lengths[minlen:]
    frequency = frequency[minlen:]

    if rot:
        dir_fac = 4
    else:
        dir_fac = 1

    if verbose:
        print(f"{np.sum(frequency)} vertical patches of sizes:")
        print(np.where(frequency != 0)[0] + minlen)
        print("were detected.")
        print("The recurrence rate is:", rp_sum / (ys**2 * xs**2))

    return rp_sum, lengths, frequency, dir_fac, rp_sum / (ys**2 * xs**2)


def hs_patch_sampling_fast(
    img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False
):
    """
    Function to get sizes of diagonal and vertical patches (patch sampling approach).
    """

    rp_sum, lengths_diag, frequency_diag, dir_fac_diag, RR = (
        diag_hs_patch_sampling_fast(
            img, minlen=minlen, RR=RR, epsilon=epsilon, verbose=verbose, rot=rot
        )
    )
    rp_sum, lengths_vert, frequency_vert, dir_fac_vert, RR = (
        vert_hs_patch_sampling_fast(
            img, minlen=minlen, RR=RR, epsilon=epsilon, verbose=verbose, rot=rot
        )
    )

    return (
        rp_sum,
        lengths_diag,
        frequency_diag,
        frequency_vert,
        dir_fac_diag,
        dir_fac_vert,
        RR,
    )


# %% line sampling


def get_line_lengths(img, patterns, epsilon_mod, only_upper_half=True):
    """
    Function to get the lengths and frequencies of vertical and diagonal lines
    (line sampling approach)
    """

    ys, xs = img.shape
    max_dim = max([ys, xs])

    ### initialize length and frequency arrays and recurrence plot sum

    lengths = np.arange(max_dim + 1)
    frequency_diag = np.zeros(max_dim + 1)
    frequency_vert = np.zeros(max_dim + 1)
    rp_sum = 0

    ### helper function for recurrence values of a given array of points

    def _rp(points, epsilon_mod):
        # create array to return recurrences
        rp_arr = np.zeros(points.shape[0])
        # check for valid points (points within the rp)
        idx_val = np.where(
            (np.sum(points >= 0, axis=1) == 4)
            & (np.sum(points[:, [0, 2]] < ys, axis=1) == 2)
            & (np.sum(points[:, [1, 3]] < xs, axis=1) == 2)
        )
        # calculate recurrence values for valid points, others remain 0
        rp_arr[idx_val] = (
            np.abs(
                img[points[idx_val, 0], points[idx_val, 1]]
                - img[points[idx_val, 2], points[idx_val, 3]]
            )
            <= epsilon_mod
        )
        return rp_arr.astype(bool)

    ### get line lengths

    # go through all coordinates (only upper half)
    # here it does not matter in what direction the pixels are iterated through since the rp
    # will not be adjusted
    for y1 in trange(ys):
        for x1 in range(xs):

            if only_upper_half:
                # get points from img2 (where recurrences have not yet been calculated for)
                yy, xx = np.meshgrid(np.arange(y1, ys), np.arange(xs))
                idx = (yy.flatten() != y1) | (x1 < xx.flatten())
            else:
                yy, xx = np.meshgrid(np.arange(ys), np.arange(xs))
                idx = np.ones(yy.flatten().shape[0]).astype(bool)

            # create array of potential line starting points
            # (where rp == 1)
            ps = np.zeros((yy.flatten()[idx].shape[0], 4))
            ps[:, 0] = y1
            ps[:, 1] = x1
            ps[:, 2] = yy.flatten()[idx]
            ps[:, 3] = xx.flatten()[idx]
            ps = ps.astype(int)
            ps = ps[_rp(ps, epsilon_mod)]
            xx = []
            yy = []

            rp_sum += ps.shape[0]

            # continue if there are no potential starting points
            if ps.size == 0:
                continue

            # create points and directions arrays
            p = np.repeat(ps, patterns.shape[0], axis=0)
            k = np.vstack(ps.shape[0] * [patterns])
            ps = []

            # check surroundings in all directions and only take those directions
            # where p-k is 0 (only then beginning of diagonal)
            idx = ~_rp(p - k, epsilon_mod)
            k, p = k[idx], p[idx]

            # check if p+k is 1 to filter out single pixels that are not the
            # beginning of a longer line structure
            if k.size != 0:
                idx = _rp(p + k, epsilon_mod)
                k, p = k[idx], p[idx]

            # run through all lines while increasing the length,
            # stop in individual direction when line ends
            l = 2
            while k.size != 0:
                ring_rp = _rp(p + k * l, epsilon_mod)
                np.add.at(
                    frequency_vert, l, np.sum(~ring_rp[(k[:, 0] == 0) & (k[:, 1] == 0)])
                )
                np.add.at(
                    frequency_diag, l, np.sum(~ring_rp[(k[:, 0] != 0) | (k[:, 1] != 0)])
                )
                k, p = k[ring_rp], p[ring_rp]
                l += 1

    return rp_sum, lengths, frequency_vert, frequency_diag


def diag_hs_line_sampling_fast(
    img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False
):
    """
    Get only the diagonal lines (line sampling approach).
    """

    ys, xs = img.shape

    ### define rotation patterns

    # the rotation patterns are the factors for k1, k2, k3 and k4
    # rotation only in 45° steps
    # (k1, k2) != (0, 0) and (k3, k4) != (0, 0)
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

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    ### get line length distributions

    rp_sum, lengths, _, frequency = get_line_lengths(img, patterns, epsilon_mod)

    ### account for only evaluating the upper half and half of the directionsS

    # double the frequency bc only upper half
    # BEFORE adding the main diagonal
    frequency = frequency * 2

    # adjust the rp sum (upper half)
    rp_sum = rp_sum * 2

    # add the main diagonal
    if rot:
        # add the main HS sampled by diagonal lines
        frequency[list(range(minlen, np.min([xs, ys]) + 1))] += 2
        frequency[list(range(minlen, np.min([xs, ys])))] += 2
        frequency[np.min([ys, xs])] += 2 * np.abs(ys - xs)

        # add the main HS sampled by straight lines
        frequency[ys] += xs
        frequency[xs] += ys

        # adjust recurrence plot sum
        rp_sum += ys * xs

        # adjust frequency (mirroring directions)
        frequency *= 2

        # set direction factor for rotation
        dir_fac = 64
    else:
        # add the main HS sampled by diagonal lines
        frequency[list(range(minlen, np.min([xs, ys]) + 1))] += 1
        frequency[list(range(minlen, np.min([xs, ys])))] += 1
        frequency[np.min([ys, xs])] += np.abs(ys - xs)

        # add the main HS sampled by straight lines
        frequency[ys] += xs

        rp_sum += ys * xs

        dir_fac = 4

    # only return the lengths >= minlen
    lengths = lengths[minlen:]
    frequency = frequency[minlen:]

    if verbose:
        print(f"{np.sum(frequency)} diagonal lines of lenghts:")
        print(np.where(frequency != 0)[0] + minlen)
        print("were detected.")
        print("The recurrence rate is:", rp_sum / (ys**2 * xs**2))

    return rp_sum, lengths, frequency, dir_fac, rp_sum / (ys**2 * xs**2)


def vert_hs_line_sampling_fast(
    img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False
):
    """
    Get only the vertical lines (line sampling approach).
    """

    ys, xs = img.shape

    ### define rotation patterns

    # the rotation patterns are the factors for k1, k2, k3 and k4
    # rotation only in 45° steps
    # (k1, k2) != (0, 0) and (k3, k4) != (0, 0)
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
    patterns = np.array([(0, 0, i, j) for i, j in rot_pattern])

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

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    ### get line length distributions

    print(f"Detecting vertical lines in {len(rot_pattern)} directions...")
    rp_sum, lengths, frequency, _ = get_line_lengths(
        img, patterns, epsilon_mod, only_upper_half=False
    )

    ### account for only evaluating the upper half and half of the directions

    # double the frequency bc only upper half
    # BEFORE adding the main diagonal
    #### frequency = frequency * 2

    # adjust the rp sum (upper half)
    #### rp_sum = rp_sum * 2

    # add the main diagonal
    if rot:
        dir_fac = 8
        frequency *= 2  # (mirroring dirs)
        #### rp_sum += ys * xs
    else:
        dir_fac = 2
        #### rp_sum += ys * xs

    # only return the lengths >= minlen
    lengths = lengths[minlen:]
    frequency = frequency[minlen:]

    if verbose:
        print(f"{np.sum(frequency)} vertical lines of lenghts:")
        print(np.where(frequency != 0)[0] + minlen)
        print("were detected.")
        print("The recurrence rate is:", rp_sum / (ys**2 * xs**2))

    return rp_sum, lengths, frequency, dir_fac, rp_sum / (ys**2 * xs**2)


def hs_line_sampling_fast(img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False):
    """
    Get diagonal and vertical lines (line sampling approach).
    """

    ys, xs = img.shape

    ### define rotation patterns

    # the rotation patterns are the factors for k1, k2, k3 and k4
    # rotation only in 45° steps
    # (k1, k2) != (0, 0) and (k3, k4) != (0, 0)
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

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    ### get line length distributions

    rp_sum, lengths, frequency_vert, frequency_diag = get_line_lengths(
        img, patterns, epsilon_mod
    )

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
            print(np.where(frequency_diag != 0)[0] + minlen)
            print("were detected.")
        except:
            print("No diagonal lines detected.")
        try:
            print(f"{np.sum(frequency_vert)} vertical lines of lenghts:")
            print(np.where(frequency_vert != 0)[0] + minlen)
            print("were detected.")
        except:
            print("No vertical lines detected.")
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


def get_mass_distribution_memory(img, ys, xs, epsilon_mod, max_box_size):
    """
    Slower, but more memory efficient.

    Idea:
    For each box size, until the maximum box size, increase the size of the hyper-cube with the
    top left corner being the current pixel, with each iteration. This is done by adding
    an additional hyper-cube layer with each iteration.
    """

    mass_distribution = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_z2 = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse_z2 = np.zeros((max_box_size + 1), dtype=np.int64)

    for y1 in trange(ys):
        for x1 in range(xs):
            for y2 in range(ys):
                for x2 in range(xs):

                    # check the maximum box size that really needs to be calculated
                    new_max_box_size = np.where(np.arange(max_box_size + 1) % 1 == 0)[
                        0
                    ][-1]

                    m = 0  # initial mass
                    for w in range(1, new_max_box_size + 1):

                        if (
                            (y1 + w > ys)
                            or (x1 + w > xs)
                            or (y2 + w > ys)
                            or (x2 + w > xs)
                        ):
                            break

                        # add the 3d neighbouring hyper surfaces
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 : y1 + w, x1 : x1 + w, np.newaxis]
                                    - img[y2 : y2 + w, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 : y1 + w, x1 : x1 + w, np.newaxis]
                                    - img[y2 + w - 1, x2 : x2 + w]
                                )
                                <= epsilon_mod
                            )
                        )
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 : y1 + w, x1 + w - 1]
                                    - img[y2 : y2 + w, x2 : x2 + w, np.newaxis]
                                )
                                <= epsilon_mod
                            )
                        )
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 : x1 + w]
                                    - img[y2 : y2 + w, x2 : x2 + w, np.newaxis]
                                )
                                <= epsilon_mod
                            )
                        )

                        # subtract the 2d surfaces (that were counted double)
                        m -= int(
                            np.sum(
                                abs(
                                    img[y1 : y1 + w, x1 : x1 + w]
                                    - img[y2 + w - 1, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )
                        m -= int(
                            np.sum(
                                abs(
                                    img[y1 : y1 + w, x1 + w - 1, np.newaxis]
                                    - img[y2 : y2 + w, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )
                        m -= int(
                            np.sum(
                                abs(
                                    img[y1 : y1 + w, x1 + w - 1, np.newaxis]
                                    - img[y2 + w - 1, x2 : x2 + w]
                                )
                                <= epsilon_mod
                            )
                        )
                        m -= int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 : x1 + w, np.newaxis]
                                    - img[y2 : y2 + w, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )
                        m -= int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 : x1 + w, np.newaxis]
                                    - img[y2 + w - 1, x2 : x2 + w]
                                )
                                <= epsilon_mod
                            )
                        )
                        m -= int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 + w - 1]
                                    - img[y2 : y2 + w, x2 : x2 + w]
                                )
                                <= epsilon_mod
                            )
                        )

                        # add the 1d lines (that were substracted)
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 : y1 + w, x1 + w - 1]
                                    - img[y2 + w - 1, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 : x1 + w]
                                    - img[y2 + w - 1, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 + w - 1]
                                    - img[y2 : y2 + w, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )
                        m += int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 + w - 1]
                                    - img[y2 + w - 1, x2 : x2 + w]
                                )
                                <= epsilon_mod
                            )
                        )

                        # substract corner point (that was counted multiple times)
                        m -= int(
                            np.sum(
                                abs(
                                    img[y1 + w - 1, x1 + w - 1]
                                    - img[y2 + w - 1, x2 + w - 1]
                                )
                                <= epsilon_mod
                            )
                        )

                        if y1 % w + x1 % w + y2 % w + x2 % w == 0:
                            mass_distribution[w] += m
                            mass_distribution_reverse[w] += w**4 - m
                            mass_distribution_z2[w] += m**2
                            mass_distribution_reverse_z2[w] += (w**4 - m) ** 2

    return (
        mass_distribution,
        mass_distribution_reverse,
        mass_distribution_z2,
        mass_distribution_reverse_z2,
    )


def get_mass_distribution_fast(img, ys, xs, epsilon_mod, max_box_size):
    """
    Faster, but not memory efficient. (Do not use for very large images)

    Idea:
    For each box size, until maximum box size, calculate a hyper-cube with the
    top left corner being the current pixel. Add the sum of this cube (which is recalculated
    for each box size) to the mass distribution arrays.
    """

    mass_distribution = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_z2 = np.zeros((max_box_size + 1), dtype=np.int64)
    mass_distribution_reverse_z2 = np.zeros((max_box_size + 1), dtype=np.int64)

    for y1 in trange(ys):
        for x1 in range(xs):
            for y2 in range(ys):
                for x2 in range(xs):

                    for w in range(1, max_box_size + 1):

                        if (
                            (y1 + w > ys)
                            or (x1 + w > xs)
                            or (y2 + w > ys)
                            or (x2 + w > xs)
                        ):
                            break

                        if (
                            (y1 % w != 0)
                            or (x1 % w != 0)
                            or (y2 % w != 0)
                            or (x2 % w != 0)
                        ):
                            continue

                        m = int(
                            np.sum(
                                abs(
                                    img[
                                        y1 : y1 + w, x1 : x1 + w, np.newaxis, np.newaxis
                                    ]
                                    - img[y2 : y2 + w, x2 : x2 + w]
                                )
                                <= epsilon_mod
                            )
                        )
                        mass_distribution[w] += m
                        mass_distribution_z2[w] += m**2
                        m_rev = w**4 - m
                        mass_distribution_reverse[w] += m_rev
                        mass_distribution_reverse_z2[w] += m_rev**2

    return (
        mass_distribution,
        mass_distribution_reverse,
        mass_distribution_z2,
        mass_distribution_reverse_z2,
    )


def mass_distribution_fast(
    img, epsilon=None, RR=0.2, max_box_size=None, memory_efficient=False
):
    """
    Function to get the mass distribution for an image. One less memory efficient approach and
    one more memory efficient approach are available.
    """

    assert len(img.shape) == 2, "Image needs to be 2 dimensional."
    ys, xs = img.shape

    ### set maximum box size

    if not max_box_size:
        max_box_size = int(0.1 * max(ys, xs))

    ### epsilon estimation

    if not epsilon:
        epsilon = utils.estimate_epsilon(img, RR)
    print("Epsilon:", epsilon)
    epsilon_mod = epsilon / np.sqrt(2)

    if memory_efficient:
        (
            mass_distribution,
            mass_distribution_reverse,
            mass_distribution_z2,
            mass_distribution_reverse_z2,
        ) = get_mass_distribution_memory(img, ys, xs, epsilon_mod, max_box_size)
    else:
        (
            mass_distribution,
            mass_distribution_reverse,
            mass_distribution_z2,
            mass_distribution_reverse_z2,
        ) = get_mass_distribution_fast(img, ys, xs, epsilon_mod, max_box_size)

    # for overlapping boxes
    # nr_boxes = np.hstack([np.zeros(1), ((xs - np.arange(max_box_size)) * (ys - np.arange(max_box_size))) ** 2])

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

    return (
        mass_distribution,
        mass_distribution_reverse,
        mass_distribution_z2,
        mass_distribution_reverse_z2,
        nr_boxes,
    )


def mass_distribution_2d(rp, max_box_size):
    """
    Get the mass distribution of a two-dimensional recurrence plot from
    one-dimensional data.
    """

    box_sizes = np.arange(1, max_box_size + 1)
    lac = np.zeros_like(box_sizes, dtype=np.float64)

    for bs in box_sizes:
        nr_boxes = (rp.shape[0] // bs, rp.shape[1] // bs)
        boxes = view_as_blocks(rp[: bs * nr_boxes[0], : bs * nr_boxes[1]], (bs, bs))
        boxes = boxes.reshape(
            (boxes.shape[0] * boxes.shape[1], boxes.shape[2], boxes.shape[3])
        )
        first_moment = np.sum(np.sum(boxes, axis=(1, 2))) / np.prod(nr_boxes)
        second_moment = np.sum(np.sum(boxes, axis=(1, 2)) ** 2) / np.prod(nr_boxes)
        first_moment_rev = np.sum(bs**2 - np.sum(boxes, axis=(1, 2))) / np.prod(
            nr_boxes
        )
        second_moment_rev = np.sum(
            (bs**2 - np.sum(boxes, axis=(1, 2))) ** 2
        ) / np.prod(nr_boxes)
        l = 2 - (
            first_moment**2 / second_moment + (first_moment_rev**2 / second_moment_rev)
        )
        lac[bs - 1] = l

    return box_sizes, lac
