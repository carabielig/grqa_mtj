import os

import numpy as np
import pandas as pd


def determinism_and_avg_diag_hs_size(p, l, rp_sum, dir_fac=1):
    """
    Calculate the determinism and average diagonal line length from
    the probability distribution of diagonal line lengths.
    """
    if p.size == 0:
        print("WARNING: no diagonal HS detected.")
        return 0, 0

    p = p / dir_fac

    det = np.sum(l * p) / rp_sum
    if type(det) == np.ndarray:
        det = det[0]
    l_hs = np.sum(l * p) / np.sum(p)

    return det, l_hs


def determinism_and_avg_diag_hs_size_patch(p, l, rp_sum, dir_fac=1):
    """
    Calculate the determinism and average diagonal HS size from
    the probability distribution of diagonal patch sizes.
    """
    if p.size == 0:
        print("WARNING: no diagonal HS detected.")
        return 0, 0

    p = p / dir_fac

    det = np.sum(l**2 * p) / rp_sum
    l_hs = np.sum(l * p) / np.sum(p)

    return det, l_hs


def determinism_and_avg_diag_hs_size_from_lengths(rp_sum, diag_lengths, dir_fac=1):
    """
    Calculate the determinism and average diagonal line length from
    an array containing all the line lengths of the individual diagonal lines.
    """

    if diag_lengths.size == 0:
        print("WARNING: no diaglonal HS detected.")
        return 0, 0

    p, bins = np.histogram(
        diag_lengths, bins=np.arange(np.min(diag_lengths), np.max(diag_lengths) + 2)
    )
    p = p / dir_fac
    # p = p * np.diff(bins)  # only if bin size != 1

    det = np.sum(bins[:-1] * p) / rp_sum
    l_hs = np.sum(bins[:-1] * p) / np.sum(p)

    return det, l_hs


def determinism_and_avg_diag_hs_size_patch_from_lengths(
    rp_sum, diag_lengths, dir_fac=1
):
    """
    Calculate the determinism and average diagonal HS size from
    an array containing all the patch sizes of the individual diagonal patches.
    """

    if diag_lengths.size == 0:
        print("WARNING: no diaglonal HS detected.")
        return 0, 0

    p, bins = np.histogram(
        diag_lengths, bins=np.arange(np.min(diag_lengths), np.max(diag_lengths) + 2)
    )
    p = p / dir_fac
    # p = p * np.diff(bins)  # only if bin size != 1

    det = np.sum(bins[:-1] ** 2 * p) / rp_sum
    l_hs = np.sum(bins[:-1] * p) / np.sum(p)

    return det, l_hs


def laminarity_and_trapping_size(p, l, rp_sum, dir_fac=1):
    """
    Calculate the laminarity and trapping size from
    the probability distribution of vertical line lengths.
    """
    if p.size == 0:
        print("WARNING: no vertical HS detected.")
        return 0, 0

    p = p / dir_fac

    lam = np.sum(l * p) / rp_sum
    if type(lam) == np.ndarray:
        lam = lam[0]
    tt = np.sum(l * p) / np.sum(p)

    return lam, tt


def laminarity_and_trapping_size_patch(p, l, rp_sum, dir_fac=1):
    """
    Calculate the laminarity and trapping size from
    the probability distribution of vertical patch sizes.
    """
    if p.size == 0:
        print("WARNING: no vertical HS detected.")
        return 0, 0

    p = p / dir_fac

    lam = np.sum(l**2 * p) / rp_sum
    tt = np.sum(l * p) / np.sum(p)

    return lam, tt


def laminarity_and_trapping_size_from_lengths(rp_sum, vert_lengths, dir_fac=1):
    """
    Calculate the laminarity and trapping size from
    an array containing all the line lengths of the individual vertical lines.
    """

    if vert_lengths.size == 0:
        print("WARNING: no vertical HS detected.")
        return 0, 0

    p, bins = np.histogram(
        vert_lengths, bins=np.arange(np.min(vert_lengths), np.max(vert_lengths) + 2)
    )
    p = p / dir_fac
    # p = p * np.diff(bins)  # only if bin size != 1

    lam = np.sum(bins[:-1] * p) / rp_sum
    tt = np.sum(bins[:-1] * p) / np.sum(p)

    return lam, tt


def laminarity_and_trapping_size_patch_from_lengths(rp_sum, vert_lengths, dir_fac=1):
    """
    Calculate the laminarity and trapping size from
    an array containing all the patch sizes of the individual vertical patches.
    """

    if vert_lengths.size == 0:
        print("WARNING: no vertical HS detected.")
        return 0, 0

    p, bins = np.histogram(
        vert_lengths, bins=np.arange(np.min(vert_lengths), np.max(vert_lengths) + 2)
    )
    p = p / dir_fac
    # p = p * np.diff(bins)  # only if bin size != 1

    lam = np.sum(bins[:-1] ** 2 * p) / rp_sum
    tt = np.sum(bins[:-1] * p) / np.sum(p)

    return lam, tt


def entropy(p):
    """
    Calculate the entropy from
    the probability distribution of diagonal line lengths.
    """
    if p.size == 0:
        print("WARNING: no diaglonal HS detected.")
        return np.nan

    # the number of directions (sampling of the same HS for multiple times)
    # plays no role for the relative probability
    p_rel = p / np.sum(p)

    ent = -np.nansum(p_rel * np.log(p_rel))  # correct log?

    assert ent >= 0, "Entropy cannot be negative."

    return np.abs(ent)


def entropy_from_lengths(diag_lengths):
    """
    Calculate the entropy from
    an array containing all the line lengths of the individual diagonal lines.
    """

    if diag_lengths.size == 0:
        print("WARNING: no diaglonal HS detected.")
        return np.nan

    p, bins = np.histogram(
        diag_lengths, bins=np.arange(np.min(diag_lengths), np.max(diag_lengths) + 2)
    )
    # p = p * np.diff(bins)  # only if bin size != 1

    # the number of directions (sampling of the same HS for multiple times)
    # plays no role for the relative probability
    p_rel = p / np.sum(p)

    ent = -np.nansum(p_rel * np.log(p_rel))  # correct log?

    assert ent >= 0, "Entropy cannot be negative."

    return np.abs(ent)


def lacunarity(
    mass_distribution,
    mass_distribution_reverse,
    mass_distribution_z2,
    mass_distribution_reverse_z2,
    nr_boxes,
):
    """
    Caluclate the lacunarity from the sum of all masses (mass_distribution), sum of all inverse masses
    (mass_distribution_reverse), sum of squared masses (mass_distribution_z2), and
    sum of squared inverse masses (mass_distribution_reverse_z2)
    """

    idx = np.where(nr_boxes != 0)

    first_moment = mass_distribution[idx] / nr_boxes[idx]
    first_moment_rev = mass_distribution_reverse[idx] / nr_boxes[idx]
    second_moment = mass_distribution_z2[idx] / nr_boxes[idx]
    second_moment_rev = mass_distribution_reverse_z2[idx] / nr_boxes[idx]

    # lac_normal = second_moment / first_moment**2
    # lac_rev = second_moment_rev / first_moment_rev**2

    first_moment = first_moment.astype(float)
    first_moment_rev = first_moment_rev.astype(float)
    second_moment = second_moment.astype(float)
    second_moment_rev = second_moment_rev.astype(float)

    lac_normal = np.divide(
        second_moment,
        first_moment**2,
        out=np.full_like(second_moment, np.nan, dtype=float),
        where=(first_moment**2 != 0),
    )
    lac_rev = np.divide(
        second_moment_rev,
        first_moment_rev**2,
        out=np.full_like(second_moment_rev, np.nan, dtype=float),
        where=(first_moment_rev**2 != 0),
    )

    lac = np.ones_like(nr_boxes) * np.nan
    lac[idx] = 2 - (1 / lac_normal + 1 / lac_rev)

    return lac


def lacunarity_not_normalized(
    mass_distribution,
    mass_distribution_z2,
    nr_boxes,
):
    """
    Caluclate the lacunarity (NOT NORMALIZED) from the sum of all masses (mass_distribution), sum of all inverse masses
    (mass_distribution_reverse), sum of squared masses (mass_distribution_z2), and
    sum of squared inverse masses (mass_distribution_reverse_z2)
    """

    idx = np.where(nr_boxes != 0)

    first_moment = mass_distribution[idx] / nr_boxes[idx]
    second_moment = mass_distribution_z2[idx] / nr_boxes[idx]

    lac_normal = second_moment / first_moment**2

    lac = np.ones_like(nr_boxes) * np.nan
    lac[idx] = lac_normal

    return lac


def open_grqa_measures(path, minlen=2):
    """
    Open the data frame that was saved as output from run_grqa.py. It contains the results for the
    grqa measures.
    A certain minimum length (pixel length a line needs to have to be considered relevant).
    """

    df = pd.read_csv(os.path.join(path, f"grqa_measures.csv"))
    df = df.drop(df[np.isnan(df.det_line_rot)].index)

    if minlen == 2:
        return df
    elif minlen > 2:
        minlen = int(minlen)
        for row in df.index:
            dist_df = pd.read_csv(
                os.path.join(
                    path,
                    f"{df.loc[row]["name"]}",
                    f"{df.loc[row]["name"]}_line_length_distribution.csv",
                )
            )
            diag_len = dist_df.length.to_numpy()[minlen - 2 :]
            vert_len = dist_df.length.to_numpy()[minlen - 2 :]
            diag_frequ = dist_df.diag_frequ.to_numpy()[minlen - 2 :]
            vert_frequ = dist_df.vert_frequ.to_numpy()[minlen - 2 :]
            diag_rp_sum = eval(dist_df.rp_sum[0])[0]
            vert_rp_sum = eval(dist_df.rp_sum[0])[0]
            diag_dir_fac, vert_dir_fac = 4, 2

            df.loc[row, ["det_line", "l_hs_line"]] = determinism_and_avg_diag_hs_size(
                diag_frequ, diag_len, diag_rp_sum, dir_fac=diag_dir_fac
            )
            df.loc[row, ["lam_line", "tt_line"]] = laminarity_and_trapping_size(
                vert_frequ, vert_len, vert_rp_sum, dir_fac=vert_dir_fac
            )
            df.loc[row, "entropy_line"] = entropy(diag_frequ)

            dist_df = pd.read_csv(
                os.path.join(
                    path,
                    f"{df.loc[row]["name"]}",
                    f"{df.loc[row]["name"]}_line_length_distribution_rot.csv",
                )
            )
            diag_len = dist_df.length.to_numpy()[minlen - 2 :]
            vert_len = dist_df.length.to_numpy()[minlen - 2 :]
            diag_frequ = dist_df.diag_frequ.to_numpy()[minlen - 2 :]

            vert_frequ = dist_df.vert_frequ.to_numpy()[minlen - 2 :]
            diag_rp_sum = eval(dist_df.rp_sum[0])[0]
            vert_rp_sum = eval(dist_df.rp_sum[0])[0]
            diag_dir_fac, vert_dir_fac = 64, 8

            df.loc[row, ["det_line_rot", "l_hs_line_rot"]] = (
                determinism_and_avg_diag_hs_size(
                    diag_frequ, diag_len, diag_rp_sum, dir_fac=diag_dir_fac
                )
            )
            df.loc[row, ["lam_line_rot", "tt_line_rot"]] = laminarity_and_trapping_size(
                vert_frequ, vert_len, vert_rp_sum, dir_fac=vert_dir_fac
            )
            df.loc[row, "entropy_line_rot"] = entropy(diag_frequ)
            return df
    else:
        print("Invalid minlen.")
        return []
