import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm, trange


def distance_matrix(greyscale):
    """
    Get the distance matrix of an image using pairwise euclidean distance.
    To get the "real" distances, divide by sqrt(2).
    """

    greyscale = np.ascontiguousarray(greyscale)

    dm = squareform(
        pdist(np.c_[greyscale.flatten(), greyscale.flatten()], metric="euclidean")
    )
    dm = dm.reshape(
        (greyscale.shape[0], greyscale.shape[1], greyscale.shape[0], greyscale.shape[1])
    )

    return dm


def recurrence_threshold(distance_matrix, RR=0.2):
    """
    Get the recurrence threshold for a defined recurrence rate RR.
    """

    distance_matrix = distance_matrix.flatten()
    epsilon = np.quantile(distance_matrix, RR)

    return epsilon


def recurrence_plot(distance_matrix, epsilon):
    """
    Get the recurrence plot with a specified epsilon value.
    """

    rp = np.zeros(distance_matrix.shape)
    rp[distance_matrix <= epsilon] = 1
    rp = rp.astype(np.int32)

    return rp


def diag_hs_line_sampling(
    rp, minlen=2, verbose=True, rot=False, return_distribution=True
):
    """
    Very slow line sampling approach that also saves the indices
    of the start of the line. Used for small images and testing.
    """

    # get the rotation patterns
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

    indices = []
    lengths = []
    directions = []

    i = 0

    ys, xs = rp.shape[:2]

    print(f"Detecting diagonal lines in {len(rot_pattern)**2} directions...")

    # go through all combinations of rotation patterns
    for rot_p1 in tqdm(rot_pattern):
        k1 = rot_p1[0]
        k2 = rot_p1[1]
        for rot_p2 in rot_pattern:
            k3 = rot_p2[0]
            k4 = rot_p2[1]

            # go through all coordinates
            for y1 in range(rp.shape[0]):
                for x1 in range(rp.shape[1]):
                    for y2 in range(rp.shape[2]):
                        for x2 in range(rp.shape[3]):
                            # check if the pixel is 1
                            # (otherwise it can't be on a line)
                            if not rp[y1, x1, y2, x2] == 1:
                                continue

                            # check if the pixel is not on the edge or of the pixel on the opposite direction is not 0
                            # if pixel is e.g. on the right edge and bottom left pixel is checked, it would still
                            # only be l = 1 which is < lmin
                            if (
                                (0 <= y1 - k1 < ys)
                                and (0 <= x1 - k2 < xs)
                                and (0 <= y2 - k3 < ys)
                                and (0 <= x2 - k4 < xs)
                                and rp[y1 - k1, x1 - k2, y2 - k3, x2 - k4] != 0
                            ):
                                continue

                            # go along vertical line to find the length
                            l = -1
                            d = 1

                            while d == 1:
                                l += 1

                                # catch index error to avoid the shapes leaving top or right side
                                try:
                                    d = rp[
                                        y1 + k1 * l,
                                        x1 + k2 * l,
                                        y2 + k3 * l,
                                        x2 + k4 * l,
                                    ]
                                except IndexError:
                                    d = 0

                                # to avoid that the shapes leave the left or bottom side of the rp
                                if (
                                    sum(
                                        np.array(
                                            [
                                                y1 + k1 * l,
                                                x1 + k2 * l,
                                                y2 + k3 * l,
                                                x2 + k4 * l,
                                            ]
                                        )
                                        < 0
                                    )
                                    != 0
                                ):
                                    d = 0

                            if l >= minlen:
                                indices.append([y1, x1, y2, x2])
                                lengths.append(l)
                                directions.append([k1, k2, k3, k4])
                                i += 1

    lengths = np.array(lengths)
    indices = np.array(indices)
    directions = np.array(directions)

    if rot:
        dir_fac = 64
    else:
        dir_fac = 4

    if verbose:
        print(f"{len(lengths)} diagonal lines of lenghts:")
        print(np.unique(np.array(lengths)))
        print("were detected.")

    if return_distribution:
        unique, counts = np.unique(lengths, return_counts=True)

        lengths = np.arange(np.max([ys, xs]) + 1)
        frequency = np.zeros(np.max([ys, xs]) + 1)

        frequency[unique] += counts

        lengths = lengths[minlen:]
        frequency = frequency[minlen:]

        return np.sum(rp), lengths, frequency, dir_fac
    else:
        return (
            np.sum(rp),
            np.array(indices),
            np.array(lengths),
            np.array(directions),
            dir_fac,
        )


def vert_hs_line_sampling(
    rp, minlen=2, verbose=True, rot=False, return_distribution=True
):
    """
    Very slow line sampling approach that also saves the indices
    of the start of the line. Used for small images and testing.
    """

    # get the rotation patterns
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

    indices = []
    lengths = []
    directions = []

    i = 0

    ys, xs = rp.shape[:2]

    print(f"Detecting vertical lines in {len(rot_pattern)} directions...")

    # go through all rotation patterns
    for rot_p in tqdm(rot_pattern):
        k1 = rot_p[0]
        k2 = rot_p[1]

        # go through all coordinates
        for y1 in range(rp.shape[0]):
            for x1 in range(rp.shape[1]):
                for y2 in range(rp.shape[2]):
                    for x2 in range(rp.shape[3]):

                        # check if the pixel is 1
                        # (otherwise it can't be on a line)
                        if not rp[y1, x1, y2, x2] == 1:
                            continue

                        if (
                            (0 <= y2 - k1 < ys)
                            and (0 <= x2 - k2 < xs)
                            and rp[y1, x1, y2 - k1, x2 - k2] != 0
                        ):
                            continue

                        # go along vertical line to find the length
                        l = -1
                        d = 1

                        while d == 1:
                            l += 1

                            # catch index error to avoid the shapes leaving top or right side
                            try:
                                d = rp[y1, x1, y2 + k1 * l, x2 + k2 * l]
                            except IndexError:
                                d = 0

                            # to avoid that the shapes leave the left or bottom side of the rp
                            if (
                                sum(np.array([y1, x1, y2 + k1 * l, x2 + k2 * l]) < 0)
                                != 0
                            ):
                                d = 0

                        if l >= minlen:
                            indices.append([y1, x1, y2, x2])
                            lengths.append(l)
                            directions.append([k1, k2])
                            i += 1

    lengths = np.array(lengths)
    indices = np.array(indices)
    directions = np.array(directions)

    if rot:
        dir_fac = 8
    else:
        dir_fac = 2

    if verbose:
        print(f"{len(lengths)} vertical lines of lenghts:")
        print(np.unique(np.array(lengths)))
        print("were detected.")

    if return_distribution:
        unique, counts = np.unique(lengths, return_counts=True)

        lengths = np.arange(np.max([ys, xs]) + 1)
        frequency = np.zeros(np.max([ys, xs]) + 1)

        frequency[unique] += counts

        lengths = lengths[minlen:]
        frequency = frequency[minlen:]

        return np.sum(rp), lengths, frequency, dir_fac
    else:
        return (
            np.sum(rp),
            np.array(indices),
            np.array(lengths),
            np.array(directions),
            dir_fac,
        )


def diag_hs_patch_sampling(
    rp, minlen=2, verbose=True, rot=False, return_distribution=True
):
    """
    Very slow patch sampling approach that also saves the indices
    of the start of the patch. Used for small images and testing.
    """

    # get the rotation patterns
    if not rot:
        rot_pattern = [(1, 1)]
    else:
        rot_pattern = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    indices = []
    lengths = []
    directions = []

    rp_whole = rp.copy()

    i = 0

    ys, xs = rp.shape[:2]

    print(f"Detecting diagonal HS in {len(rot_pattern)**2} direction(s)...")
    for rot_p1 in tqdm(rot_pattern):
        k1 = rot_p1[0]
        k2 = rot_p1[1]

        # for each direction the direction of the iteration through the coordinates
        # has to be changed, so that the HS get detected from the correct side
        if k1 < 0:
            yr1 = range(rp.shape[0] - 1, -1, k1)
        else:
            yr1 = range(rp.shape[0])

        if k2 < 0:
            xr1 = range(rp.shape[1] - 1, -1, k2)
        else:
            xr1 = range(rp.shape[1])

        for rot_p2 in rot_pattern:
            k3 = rot_p2[0]
            k4 = rot_p2[1]

            if k3 < 0:
                yr2 = range(rp.shape[0] - 1, -1, k3)
            else:
                yr2 = range(rp.shape[0])

            if k4 < 0:
                xr2 = range(rp.shape[1] - 1, -1, k4)
            else:
                xr2 = range(rp.shape[1])

            rp = rp_whole.copy()

            # go through all coordinates
            for y1 in yr1:
                for x1 in xr1:
                    for y2 in yr2:
                        for x2 in xr2:

                            # check if the pixel is 1
                            # (otherwise it can't be on a HS)
                            if not rp[y1, x1, y2, x2] == 1:
                                continue

                            # go along diagonal to find the length
                            l = -1
                            d = 1

                            while d == 1:
                                l += 1

                                # catch index error if next pixel is outside the plot
                                try:
                                    d = rp[
                                        y1 + k1 * l,
                                        x1 + k2 * l,
                                        y2 + k3 * l,
                                        x2 + k4 * l,
                                    ]
                                except IndexError:
                                    d = 0

                                # also stop if coordinates are outside the rp (on the left)
                                if (
                                    sum(
                                        np.array(
                                            [
                                                y1 + k1 * l,
                                                x1 + k2 * l,
                                                y2 + k3 * l,
                                                x2 + k4 * l,
                                            ]
                                        )
                                        < 0
                                    )
                                    != 0
                                ):
                                    d = 0

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

                                indices.append([y1, x1, y2, x2])
                                lengths.append(l)
                                directions.append([k1, k2, k3, k4])
                                i += 1

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

    if rot:
        dir_fac = 16
    else:
        dir_fac = 1

    if verbose:
        print(f"{len(lengths)} diagonal hypersurfaces of lenghts:")
        print(np.unique(np.array(lengths)))
        print("were detected.")

    if return_distribution:
        ys, xs = rp.shape[:2]

        unique, counts = np.unique(lengths, return_counts=True)

        lengths = np.arange(np.max([ys, xs]) + 1)
        frequency = np.zeros(np.max([ys, xs]) + 1)

        frequency[unique] += counts

        lengths = lengths[minlen:]
        frequency = frequency[minlen:]

        return np.sum(rp_whole), lengths, frequency, dir_fac
    else:
        return (
            np.sum(rp_whole),
            np.array(indices),
            np.array(lengths),
            np.array(directions),
            dir_fac,
        )


def vert_hs_patch_sampling(
    rp, minlen=2, verbose=True, rot=False, return_distribution=True
):
    """
    Very slow patch sampling approach that also saves the indices
    of the start of the line. Used for small images and testing.
    """

    # get the rotation patterns
    if not rot:
        rot_pattern = [(1, 1)]
    else:
        rot_pattern = [(1, 1), (-1, 1), (-1, -1), (1, -1)]

    indices = []
    lengths = []
    directions = []

    rp_whole = rp.copy()

    i = 0

    print(f"Detecting vertical HS in {len(rot_pattern)} direction(s)...")
    for rot_p in tqdm(rot_pattern):
        k1 = rot_p[0]
        k2 = rot_p[1]

        if k1 < 0:
            yr = range(rp.shape[0] - 1, -1, k1)
        else:
            yr = range(rp.shape[0])

        if k2 < 0:
            xr = range(rp.shape[1] - 1, -1, k2)
        else:
            xr = range(rp.shape[1])

        rp = rp_whole.copy()

        # go through all coordinates
        for y1 in yr:
            for x1 in xr:
                for y2 in yr:
                    for x2 in xr:

                        # check if the pixel is 1
                        # (otherwise it can't be on a HS)
                        if not rp[y1, x1, y2, x2] == 1:
                            continue

                        # go along diagonal to find the length
                        l = -1
                        d = 1

                        while d == 1:
                            l += 1

                            # catch index error if next pixel is outside the plot
                            try:
                                d = rp[y1, x1, y2 + k1 * l, x2 + k2 * l]
                            except IndexError:
                                d = 0

                            if (
                                sum(np.array([y1, x1, y2 + k1 * l, x2 + k2 * l]) < 0)
                                != 0
                            ):
                                d = 0

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

                            # remove whole HS from the rp to detect new HS, that
                            # were overlapping with this one
                            # (at this point the definition if diagonal HS is not true anymore)
                            for k1_hs in range(l):
                                for k2_hs in range(l):
                                    rp[y1, x1, y2 + k1 * k1_hs, x2 + k2 * k2_hs] = 0

                            if l >= minlen:
                                indices.append([y1, x1, y2, x2])
                                lengths.append(l)
                                directions.append([k1, k2])
                                i += 1

    if rot:
        dir_fac = 4
    else:
        dir_fac = 1

    if verbose:
        print(f"{len(lengths)} vertical hypersurfaces of lenghts:")
        print(np.unique(np.array(lengths)))
        print("were detected.")

    if return_distribution:
        ys, xs = rp.shape[:2]

        unique, counts = np.unique(lengths, return_counts=True)

        lengths = np.arange(np.max([ys, xs]) + 1)
        frequency = np.zeros(np.max([ys, xs]) + 1)

        frequency[unique] += counts

        lengths = lengths[minlen:]
        frequency = frequency[minlen:]

        return np.sum(rp_whole), lengths, frequency, dir_fac
    else:
        return (
            np.sum(rp_whole),
            np.array(indices),
            np.array(lengths),
            np.array(directions),
            dir_fac,
        )
