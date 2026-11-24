import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import tqdm
from matplotlib.lines import Line2D
from numba import cuda
from skimage.transform import downscale_local_mean
from skimage.util import view_as_blocks

from scripts import grqa, utils

parallel = True
gpu = True


def initialize_df(nr_of_tiles):
    df = pd.DataFrame()

    df["tile"] = np.arange(nr_of_tiles)

    df["det_line"] = [np.nan] * len(df)
    df["l_hs_line"] = [np.nan] * len(df)
    df["lam_line"] = [np.nan] * len(df)
    df["tt_line"] = [np.nan] * len(df)
    df["entropy_line"] = [np.nan] * len(df)
    df["rr_line"] = [np.nan] * len(df)

    return df


def process_frequencies(
    diag_rp_sum,
    diag_len,
    diag_frequ,
    vert_frequ,
    diag_dir_fac,
    vert_dir_fac,
    diag_rr,
    b,
    cm,
):
    vert_rp_sum = diag_rp_sum
    vert_len = diag_len

    dist_df = pd.DataFrame(
        {
            "length": diag_len,
            "diag_frequ": diag_frequ,
            "vert_frequ": vert_frequ,
            "diag_rp_sum": diag_rp_sum,
            "vert_rp_sum": vert_rp_sum,
        }
    )

    dist_df.to_csv(
        f"img/{cm}_{b}_line_distribution.csv",
        index=False,
    )

    det_line, l_hs_line = grqa.determinism_and_avg_diag_hs_size(
        diag_frequ, diag_len, diag_rp_sum, dir_fac=diag_dir_fac
    )
    lam_line, tt_line = grqa.laminarity_and_trapping_size(
        vert_frequ, vert_len, vert_rp_sum, dir_fac=vert_dir_fac
    )
    entropy_line = grqa.entropy(diag_frequ)
    rr_line = diag_rr

    return b, det_line, l_hs_line, lam_line, tt_line, entropy_line, rr_line


def process_slope_tile(tile, b, cm):

    (
        diag_rp_sum,
        diag_len,
        diag_frequ,
        vert_frequ,
        diag_dir_fac,
        vert_dir_fac,
        diag_rr,
    ) = grqa.hs_line_sampling_fast(tile, 2, verbose=False, rot=True, x_stepsize=1)

    b, det_line, l_hs_line, lam_line, tt_line, entropy_line, rr_line = (
        process_frequencies(
            diag_rp_sum,
            diag_len,
            diag_frequ,
            vert_frequ,
            diag_dir_fac,
            vert_dir_fac,
            diag_rr,
        )
    )

    return b, det_line, l_hs_line, lam_line, tt_line, entropy_line, rr_line


if __name__ == "__main__":

    print("Loading data frame...")
    cm_df = pd.read_csv(
        "../../data/hebungsraten.csv",
        header=None,
        names=["catchment", "rate"],
        sep=" ",
        skipinitialspace=True,
    )
    cm_df["catchment"] = cm_df["catchment"].astype(str)

    # for cm in ["4409378"]: #cm_df.catchment:
    cm = "4409378"

    # read file
    print("Loading data...")
    dataset_slp = rio.open(f"./data/slp/{cm}_lin_slp.tif")
    slp = dataset_slp.read().squeeze()

    dataset_mask = rio.open(f"./data/mask/mask_{cm}.tif")
    mask = dataset_mask.read().squeeze()

    # cut slope to mask extent
    slp[mask == 0] = np.nan
    slp = slp[:, np.nansum(slp, axis=0) != 0]
    slp = slp[np.nansum(slp, axis=1) != 0, :]
    # slp = downscale_local_mean(slp, (5, 5))

    # plot
    # plt.imshow(slp)
    # plt.show()
    # plt.close()

    print("Cutting tiles...")
    ts = 100  # tile_size
    slp_blocks = slp[: (slp.shape[0] // ts) * ts, : (slp.shape[1] // ts) * ts]
    slp_blocks = view_as_blocks(slp_blocks, (ts, ts))
    slp_blocks = slp_blocks.reshape((slp_blocks.shape[0] * slp_blocks.shape[1], ts, ts))
    slp_blocks = slp_blocks[~np.isnan(np.sum(slp_blocks, axis=(1, 2))), :, :]

    df = initialize_df(slp_blocks.shape[0])

    # size reduction for testing
    slp_blocks = slp_blocks[:2, :, :]

    if (
        parallel and gpu and cuda.is_available()
    ):  ########################################

        for b in range(slp_blocks.shape[0]):
            (
                rp_sum,
                lengths,
                frequency_diag,
                frequency_vert,
                dir_fac_diag,
                dir_fac_vert,
                rr,
            ) = grqa.hs_line_sampling_gpu(slp_blocks[b, :, :], rot=True)
            process_frequencies(
                rp_sum,
                lengths,
                frequency_diag,
                frequency_vert,
                dir_fac_diag,
                dir_fac_vert,
                rr,
                b,
                cm,
            )

    elif parallel:  ########################################

        multiprocessing.freeze_support()
        __file__ = "mtj_multiprocessing.py"

        print("Starting multiprocessing...")
        t1 = time.time()

        # https://labex.io/tutorials/python-how-to-use-concurrent-futures-in-python-452184
        results = []
        with ProcessPoolExecutor(max_workers=4) as executer:
            futures = [
                executer.submit(process_slope_tile, slp_blocks[b, :, :], b, cm)
                for b in range(slp_blocks.shape[0])
            ]

            for future in as_completed(futures):
                future = future.result()
                df.loc[future[0], ["det_line", "l_hs_line"]] = future[1], future[2]
                df.loc[future[0], ["lam_line", "tt_line"]] = future[3], future[4]
                df.loc[future[0], "entropy_line"] = future[5]
                df.loc[future[0], "rr_line"] = future[6]

        print(time.time() - t1)

    else:  ########################################

        t1 = time.time()
        for b in range(slp_blocks.shape[0]):
            _, det_line, l_hs_line, lam_line, tt_line, entropy_line, rr_line = (
                process_slope_tile(slp_blocks[b, :, :], b, cm)
            )
            df.loc[b, ["det_line", "l_hs_line"]] = det_line, l_hs_line
            df.loc[b, ["lam_line", "tt_line"]] = lam_line, tt_line
            df.loc[b, "entropy_line"] = entropy_line
            df.loc[b, "rr_line"] = rr_line
        print(time.time() - t1)

    df.to_csv(f"img/{cm}_grqa.csv", index=False)
