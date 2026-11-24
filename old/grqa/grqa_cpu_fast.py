import numpy as np
from scipy.spatial.distance import pdist
from tqdm import trange


def diag_hs_line_sampling_fast(
    img, minlen=2, RR=0.2, epsilon=None, verbose=True, rot=False
):

    # the rotation patterns are the factors for k1 and k2
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

    pattern_check = []

    ys, xs = img.shape

    lengths = np.arange(np.max([ys, xs]) + 1)
    frequency = np.zeros(np.max([ys, xs]) + 1)

    rp_sum = 0
    j = 0

    # TODO: improve
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

    def rp(y1, x1, y2, x2):
        return int(np.sqrt(2 * (img[y1, x1] - img[y2, x2]) ** 2) <= epsilon)

    print(f"Detecting diagonal lines in {len(rot_pattern)**2} directions...")

    # go through all combinations of rotation patterns
    for rot_p1 in rot_pattern:
        k1 = rot_p1[0]
        k2 = rot_p1[1]

        for rot_p2 in rot_pattern:
            k3 = rot_p2[0]
            k4 = rot_p2[1]

            # avoid the same direction
            pattern_check.append([-k1, -k2, -k3, -k4])
            if ([k1, k2, k3, k4] in pattern_check) and rot:
                print("Skip direction...")
                continue

            # go through all coordinates (only upper half)
            for y1 in trange(ys):
                for x1 in range(xs):
                    for y2 in range(y1, ys):
                        for x2 in range(xs):

                            # in the "y-line" of (y1, x1), if (y2 == y1), the x2 value has to be
                            # larger than x1 to avoid going through the same recurrences twice
                            if (y1 == y2) and (x1 >= x2):
                                continue

                            j += 1

                            # check if the pixel is 1
                            # (otherwise it can't be on a line)
                            if not rp(y1, x1, y2, x2) == 1:
                                continue

                            rp_sum += 1

                            # check if the pixel is not on the edge or of the pixel on the opposite direction is not 0
                            # if pixel is e.g. on the right edge and bottom left pixel is checked, it would still
                            # only be l = 1 which is < lmin
                            if (
                                (0 <= y1 - k1 < ys)
                                and (0 <= x1 - k2 < xs)
                                and (0 <= y2 - k3 < ys)
                                and (0 <= x2 - k4 < xs)
                                and rp(y1 - k1, x1 - k2, y2 - k3, x2 - k4) != 0
                            ):
                                continue

                            # go along diagonal line to find the length
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
                                    d = rp(
                                        y1 + k1 * l,
                                        x1 + k2 * l,
                                        y2 + k3 * l,
                                        x2 + k4 * l,
                                    )

                            if l >= minlen:
                                frequency[l] += 1

    # double the frequency bc only upper half
    # BEFORE adding the main diagonal
    frequency = frequency * 2

    # adjust the rp sum (upper half)
    rp_sum = rp_sum * 2

    if rot:
        # add the main HS sampled by diagonal lines
        frequency[list(range(minlen, np.min([xs, ys]) + 1))] += 2
        frequency[list(range(minlen, np.min([xs, ys])))] += 2
        frequency[np.min([ys, xs])] += 2 * np.abs(ys - xs)

        # add the main HS sampled by straight lines
        frequency[ys] += xs
        frequency[xs] += ys

        rp_sum /= 32
        rp_sum += ys * xs
        frequency *= 2  # for the mirrored directions

        dir_fac = 64
    else:
        # add the main HS sampled by diagonal lines
        frequency[list(range(minlen, np.min([xs, ys]) + 1))] += 1
        frequency[list(range(minlen, np.min([xs, ys])))] += 1
        frequency[np.min([ys, xs])] += np.abs(ys - xs)

        # add the main HS sampled by straight lines
        frequency[ys] += xs

        rp_sum /= 4
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

    # the rotation pattern are the factors for k1 and k2
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

    pattern_check = []

    ys, xs = img.shape

    lengths = np.arange(np.max([ys, xs]) + 1)
    frequency = np.zeros(np.max([ys, xs]) + 1)

    i = 0
    rp_sum = 0

    # TODO: improve
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

    def rp(y1, x1, y2, x2):
        return int(np.sqrt(2 * (img[y1, x1] - img[y2, x2]) ** 2) <= epsilon)

    print(f"Detecting vertical lines in {len(rot_pattern)} directions...")

    # go through all rotation patterns
    for rot_p in rot_pattern:
        k1 = rot_p[0]
        k2 = rot_p[1]

        pattern_check.append([-k1, -k2])
        if ([k1, k2] in pattern_check) and rot:
            print("Skip direction...")
            continue

        # go through all coordinates (only upper half)
        for y1 in trange(ys):
            for x1 in range(xs):
                for y2 in range(ys):  #####range(y1, ys):
                    for x2 in range(xs):

                        # in the "y-line" of (y1, x1), if (y2 == y1), the x2 value has to be
                        # larger than x1 to avoid going through the same recurrences twice
                        ##### if (y1 == y2) and (x1 >= x2):
                        #####    continue

                        # check if the pixel is 1
                        # (otherwise it can't be on a line)
                        if not rp(y1, x1, y2, x2) == 1:
                            continue

                        rp_sum += 1

                        # check if the pixel is not on the edge or of the pixel on the opposite direction is not 0
                        # if pixel is e.g. on the right edge and bottom left pixel is checked, it would still
                        # only be l = 1 which is < lmin
                        if (
                            (0 <= y2 - k1 < ys)
                            and (0 <= x2 - k2 < xs)
                            and rp(y1, x1, y2 - k1, x2 - k2) != 0
                        ):
                            continue

                        # go along vertical line to find the length
                        l = -1
                        d = 1

                        while d == 1:
                            l += 1

                            if not (
                                (0 <= y2 + k1 * l < ys) and (0 <= x2 + k2 * l < xs)
                            ):
                                d = 0
                            else:
                                d = rp(y1, x1, y2 + k1 * l, x2 + k2 * l)

                        if l >= minlen:
                            frequency[l] += 1

    # double the frequency bc only upper half
    ##### frequency = frequency * 2
    # there is no main HS that needs to be added here, because there is none in the vertical direction
    # also, vertical HS can't cross the center

    # adjust the rp sum
    ##### rp_sum *= 2

    # only return the lengths >= minlen
    lengths = lengths[minlen:]
    frequency = frequency[minlen:]

    if rot:
        dir_fac = 8
        frequency *= 2
        rp_sum /= 4
        ##### rp_sum += ys * xs
    else:
        dir_fac = 2
        rp_sum /= 2
        ##### rp_sum += ys * xs

    if verbose:
        print(f"{np.sum(frequency)} vertical lines of lenghts:")
        print(np.where(frequency != 0)[0] + minlen)
        print("were detected.")
        print("The recurrence rate is:", rp_sum / (ys**2 * xs**2))

    return rp_sum, lengths, frequency, dir_fac, rp_sum / (ys**2 * xs**2)


