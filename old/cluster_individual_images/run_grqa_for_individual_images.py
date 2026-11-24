import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from scripts import grqa, utils
import pandas as pd
import time
from numba import cuda


### functions


def initialize_df(name_list):
    df = pd.DataFrame()

    df["name"] = name_list
    df["epsilon"] = [np.nan] * len(df)

    df["det_line"] = [np.nan] * len(df)
    df["l_hs_line"] = [np.nan] * len(df)
    df["lam_line"] = [np.nan] * len(df)
    df["tt_line"] = [np.nan] * len(df)
    df["entropy_line"] = [np.nan] * len(df)
    df["rr_line"] = [np.nan] * len(df)

    df["det_line_rot"] = [np.nan] * len(df)
    df["l_hs_line_rot"] = [np.nan] * len(df)
    df["lam_line_rot"] = [np.nan] * len(df)
    df["tt_line_rot"] = [np.nan] * len(df)
    df["entropy_line_rot"] = [np.nan] * len(df)
    df["rr_line_rot"] = [np.nan] * len(df)

    return df


def plot_and_save(
    pattern,
    epsilon,
    pattern_title,
    path="img/test_patterns/",
    real_title="",
    cmap="viridis",
):

    if len(pattern_title.split(" ")) > 1:
        pattern_name = (("_").join(pattern_title.split(" "))).lower()
    else:
        pattern_name = pattern_title.lower()

    if not real_title:
        real_title = pattern_title

    utils.plot_rp_3d(pattern, epsilon, plot_path=path, name=f"{pattern_name}_3drp")

    fig, ax = plt.subplots(1, 1, figsize=(5, 6))

    cb = ax.imshow(pattern, cmap=cmap, origin="lower")
    ax.set_title(real_title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)

    plt.colorbar(cb, fraction=0.046, ax=ax)
    plt.savefig(f"{os.path.join(path, pattern_name)}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "arr_file",
        type=str,
        help="Path to where the array (and other data) is stored + filename",
    )
    parser.add_argument("out_path", type=str, help="Path to the output directory")
    parser.add_argument(
        "RR", type=float, help="Recurrence Rate for the Recurrence Plot"
    )
    parser.add_argument(
        "max_box_size",
        type=int,
        help="Maximum box size for lacunarity (with the step size 1)",
    )
    parser.add_argument(
        "max_box_size_large",
        type=int,
        help="Maximum box size for lacunarity (with the step size box_size_store)",
    )
    parser.add_argument(
        "box_size_store",
        type=int,
        help="Box size for lacunarity which gets stored to calculate lacunarity for larger box sizes",
    )

    args = parser.parse_args()

    ### check if cuda is available

    if cuda.is_available():
        print("INFO: GPU available.")
        line_func = grqa.hs_line_sampling_gpu
        mass_func = grqa.mass_distribution_gpu_large
        mass_func_args = {
            "max_box_size_large": args.max_box_size_large,
            "box_size_store": args.box_size_store,
        }
    else:
        print("INFO: GPU not available.")
        line_func = grqa.hs_line_sampling_fast
        mass_func = grqa.mass_distribution_fast
        mass_func_args = {}

    t1 = time.time()

    ### get filepath

    arrpath = args.arr_file
    metadata_path = (args.arr_file).split(".")[0] + ".csv"

    ### load files

    img = np.load(arrpath)
    metadata = pd.read_csv(metadata_path)
    name = str(metadata.name[0])
    epsilon = metadata.epsilon[0]

    filepath = os.path.join(args.out_path, name)
    if not os.path.isdir(filepath):
        os.mkdir(filepath)

    ### initialize grqa measure df

    df = initialize_df([name])

    ### estimate epsilon

    if (not isinstance(epsilon, float)) or (np.isnan(epsilon)):
        epsilon_arr = np.zeros(10)
        for i in range(epsilon_arr.size):
            epsilon_arr[i] = utils.estimate_epsilon(img, args.RR, mode="random")
        epsilon = epsilon_arr.mean()

    ### plot and save

    plot_and_save(img, epsilon, name, path=filepath)

    ### line length distribution

    for rot in [True, False]:
        rp_sum, lengths, diag_frequ, vert_frequ, diag_dir_fac, vert_dir_fac, rr = (
            line_func(img, epsilon=epsilon, rot=rot)
        )

        if rot:
            rot_str = "_rot"
        else:
            rot_str = ""

        df.loc[0, "epsilon"] = epsilon

        df.loc[0, [f"det_line{rot_str}", f"l_hs_line{rot_str}"]] = (
            grqa.determinism_and_avg_diag_hs_size(
                diag_frequ, lengths, rp_sum, dir_fac=diag_dir_fac
            )
        )
        df.loc[0, [f"lam_line{rot_str}", f"tt_line{rot_str}"]] = (
            grqa.laminarity_and_trapping_size(
                vert_frequ, lengths, rp_sum, dir_fac=vert_dir_fac
            )
        )
        df.loc[0, f"entropy_line{rot_str}"] = grqa.entropy(diag_frequ)
        df.loc[0, f"rr_line{rot_str}"] = rr

        dist_df = pd.DataFrame(
            {
                "length": lengths,
                "diag_frequ": diag_frequ,
                "vert_frequ": vert_frequ,
                "rp_sum": [rp_sum] * len(lengths),
            }
        )
        dist_df.to_csv(
            f"{filepath}/{df.loc[0]["name"]}_line_length_distribution{rot_str}.csv",
            index=False,
        )

    ### mass distribution

    # mass_distribution, mass_distribution_reverse, mass_distribution_z2, mass_distribution_reverse_z2, nr_boxes = mass_func(img, epsilon=epsilon, max_box_size=args.max_box_size)
    (
        mass_distribution,
        mass_distribution_reverse,
        mass_distribution_z2,
        mass_distribution_reverse_z2,
        nr_boxes,
    ) = mass_func(
        img, epsilon=epsilon, max_box_size=args.max_box_size, **mass_func_args
    )

    dist_df = pd.DataFrame(
        {
            "box_size": np.arange(args.max_box_size_large + 1),
            "nr_boxes": nr_boxes,
            "mass_distribution": mass_distribution,
            "mass_distribution_reverse": mass_distribution_reverse,
            "mass_distribution_z2": mass_distribution_z2,
            "mass_distribution_reverse_z2": mass_distribution_reverse_z2,
            "lacunarity": grqa.lacunarity(
                mass_distribution,
                mass_distribution_reverse,
                mass_distribution_z2,
                mass_distribution_reverse_z2,
                nr_boxes,
            ),
        }
    )
    dist_df.to_csv(f"{filepath}/{df.loc[0]["name"]}_mass_distribution.csv", index=False)

    df.to_csv(os.path.join(filepath, "grqa_measures.csv"), index=False)

    print("TOTAL TIME:", time.time() - t1)
