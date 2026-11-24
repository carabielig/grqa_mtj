from scripts import grqa, utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from numba import cuda
import time
from skimage.transform import downscale_local_mean

### set variables

# dtm/slp
#max_box_size = 50
#max_box_size_large = 300
#box_size_store = 20
#RR = 0.2
#nthreads = (16, 16)
#resolution = 1 #m
#epsilon_mode = "random"
#rot_list = [True]
#estimate_variability = False
#variability_step_size = None

# path = f"/p/tmp/carabi/dtm_{resolution}m/"
# test_images, epsilon_list, name_list = utils.collect_dtm("/p/tmp/carabi/data/dtm_lir")

#path = f"/p/tmp/carabi/slp_{resolution}m/"
#test_images, epsilon_list, name_list = utils.collect_slp("/p/tmp/carabi/data/slp_lir")

#path = "/p/tmp/carabi/white_noise"
#test_images, epsilon_list, name_list = utils.collect_white_noise("/p/tmp/carabi/data/white_noise")

# bacteria
#max_box_size = 50
#max_box_size_large = 250
#box_size_store = 10
#RR = 0.2
#nthreads = (16, 16)
#resolution = 1
#epsilon_mode = "random"
#rot_list = [True]
#estimate_variability = False
#variability_step_size = None

#path = "/p/tmp/carabi/bacteria_images_smooth/"
#test_images, epsilon_list, name_list = utils.collect_bacteria_images("/p/tmp/carabi/data/bacteria_images", smooth=True)

# test images
#max_box_size = 100 #10 #100
#max_box_size_large = 100
#box_size_store = 100 #5 #100
#RR = 0.2
#nthreads = (16, 16)
#resolution = 1
#epsilon_mode = "whole"
#rot_list = [True, False]
#estimate_variability = False
#variability_step_size = None

#path = "/p/tmp/carabi/test_images/"
#test_images, epsilon_list, name_list = utils.collect_test_images("/p/tmp/carabi/data/test_images")

# checkerboard
max_box_size = 20
max_box_size_large = 20
box_size_store = 20
RR = 0.2
nthreads = (16, 16)
resolution = 1
epsilon_mode = "whole"
rot_list = [True]
estimate_variability = False
variability_step_size = None

path = "/p/tmp/carabi/checkerboard"
test_images, epsilon_list, name_list = utils.collect_checkerboard("/p/tmp/carabi/data/checkerboard")


### print all variables

print("max_box_size:", max_box_size)
print("max_box_size_large:", max_box_size_large)
print("box_size_store:", box_size_store)
print("RR:", RR)
print("nthreads:", nthreads)
print("resolution:", resolution)
print("epsilon mode:", epsilon_mode)
print("rot list:", rot_list)
print("estimate variability:", estimate_variability)
print("variability step size:", variability_step_size)


print("path", path)


### check if cuda is available

if cuda.is_available():
    print("INFO: GPU available.")
    line_func = grqa.hs_line_sampling_gpu
    line_func_args = {
        "nthreads": nthreads,
    }
    mass_func = grqa.mass_distribution_gpu_large
    mass_func_args = {
        "nthreads": nthreads,
        "max_box_size_large": max_box_size_large,
        "box_size_store": box_size_store,
    }
else:
    print("INFO: GPU not available.")
    line_func = grqa.hs_line_sampling_fast
    line_func_args = {}
    mass_func = grqa.mass_distribution_fast
    mass_func_args = {}


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


# %%

if __name__ == "__main__":

    print("Starting script...")

    t1 = time.time()

    ### initialize grqa measure df

    df = initialize_df(name_list)

    for i in range(len(test_images)):

        print(f"Catchment #{i}: {name_list[i]}")
        print(f"Shape: {test_images[i].shape}")

        try:

            ### cut array to GPU conform shape

            arr = test_images[i]
            if resolution > 1:
                arr = downscale_local_mean(arr, (resolution, resolution))
            orig = arr.copy()
            ys_new, xs_new = utils.get_new_shape(
                arr.shape[0], arr.shape[1], nthreads=nthreads
            )
            arr = arr[
                arr.shape[0] // 2 - ys_new // 2 : arr.shape[0] // 2 + ys_new // 2,
                arr.shape[1] // 2 - xs_new // 2 : arr.shape[1] // 2 + xs_new // 2,
            ]
            print(f"New Shape: {arr.shape}")

            ### set filepath

            filepath = os.path.join(path, name_list[i])
            if not os.path.isdir(filepath):
                os.mkdir(filepath)

            print(filepath)

            ### skip if mass distribution exists already

            #if os.path.isfile(f"{filepath}/{df.loc[i]["name"]}_mass_distribution.csv"):
            #   print("Mass distribution has been calculated already, skip...")
            #   continue

            ### estimate epsilon

            epsilon = epsilon_list[i]
            if (not epsilon) or np.isnan(epsilon):
                epsilon_arr = np.zeros(10)
                for j in range(epsilon_arr.size):
                    epsilon_arr[j] = utils.estimate_epsilon(arr, RR, mode=epsilon_mode)
                epsilon = epsilon_arr.mean()

            ### plot and save

            plot_and_save(arr, epsilon, name_list[i], path=filepath)

            ### line length distribution

            for rot in rot_list:

                (
                    rp_sum,
                    lengths,
                    diag_frequ,
                    vert_frequ,
                    diag_dir_fac,
                    vert_dir_fac,
                    rr,
                ) = line_func(arr, epsilon=epsilon, rot=rot, **line_func_args)

                if rot:
                    rot_str = "_rot"
                else:
                    rot_str = ""

                df.loc[i, "epsilon"] = epsilon

                df.loc[i, [f"det_line{rot_str}", f"l_hs_line{rot_str}"]] = (
                    grqa.determinism_and_avg_diag_hs_size(
                        diag_frequ, lengths, rp_sum, dir_fac=diag_dir_fac
                    )
                )
                df.loc[i, [f"lam_line{rot_str}", f"tt_line{rot_str}"]] = (
                    grqa.laminarity_and_trapping_size(
                        vert_frequ, lengths, rp_sum, dir_fac=vert_dir_fac
                    )
                )
                df.loc[i, f"entropy_line{rot_str}"] = grqa.entropy(diag_frequ)
                df.loc[i, f"rr_line{rot_str}"] = rr

                dist_df = pd.DataFrame(
                    {
                        "length": lengths,
                        "diag_frequ": diag_frequ,
                        "vert_frequ": vert_frequ,
                        "rp_sum": [rp_sum] * len(lengths),
                    }
                )
                dist_df.to_csv(
                    f"{filepath}/{df.loc[i]["name"]}_line_length_distribution{rot_str}.csv",
                    index=False,
                )

                if estimate_variability:

                    print("Estimating variability...")

                    if not os.path.exists(f"{filepath}/shift"):
                        os.mkdir(f"{filepath}/shift")

                    step_size = variability_step_size / resolution

                    nr_x = (orig.shape[1] - xs_new) // step_size + 1
                    nr_y = (orig.shape[0] - ys_new) // step_size + 1

                    shift_name_list = np.arange(nr_x * nr_y).astype(int).astype(str)

                    df_shift = initialize_df(shift_name_list)
                    df_shift["cm_nr"] = name_list[i]
                    df_shift["x_shift"] = len(shift_name_list) * [np.nan]
                    df_shift["y_shift"] = len(shift_name_list) * [np.nan]

                    # get all rectangles to show variance
                    shift_y = 0
                    s = 0
                    while shift_y + ys_new <= orig.shape[0]:
                        shift_x = 0
                        while shift_x + xs_new <= orig.shape[1]:

                            # cut shifted array
                            x_center = ((orig.shape[1] - xs_new) % step_size) // 2
                            y_center = ((orig.shape[0] - ys_new) % step_size) // 2
                            arr_shift = orig[
                                int(shift_y + y_center) : int(
                                    shift_y + ys_new + y_center
                                ),
                                int(shift_x + x_center) : int(
                                    shift_x + xs_new + x_center
                                ),
                            ]

                            # estimate epsilon
                            epsilon_shift = epsilon_list[i]
                            if (not epsilon_shift) or np.isnan(epsilon_shift):
                                epsilon_arr = np.zeros(10)
                                for j in range(epsilon_arr.size):
                                    epsilon_arr[j] = utils.estimate_epsilon(
                                        arr_shift, RR, mode=epsilon_mode
                                    )
                                epsilon_shift = epsilon_arr.mean()

                            (
                                rp_sum,
                                lengths,
                                diag_frequ,
                                vert_frequ,
                                diag_dir_fac,
                                vert_dir_fac,
                                rr,
                            ) = line_func(
                                arr_shift,
                                epsilon=epsilon_shift,
                                rot=rot,
                                **line_func_args
                            )

                            df_shift.loc[s, "epsilon"] = epsilon_shift

                            df_shift.loc[
                                s, [f"det_line{rot_str}", f"l_hs_line{rot_str}"]
                            ] = grqa.determinism_and_avg_diag_hs_size(
                                diag_frequ, lengths, rp_sum, dir_fac=diag_dir_fac
                            )
                            df_shift.loc[
                                s, [f"lam_line{rot_str}", f"tt_line{rot_str}"]
                            ] = grqa.laminarity_and_trapping_size(
                                vert_frequ, lengths, rp_sum, dir_fac=vert_dir_fac
                            )
                            df_shift.loc[s, f"entropy_line{rot_str}"] = grqa.entropy(
                                diag_frequ
                            )
                            df_shift.loc[s, f"rr_line{rot_str}"] = rr
                            df_shift.loc[s, f"x_shift"] = x_center + shift_x
                            df_shift.loc[s, f"y_shift"] = y_center + shift_y

                            plot_and_save(
                                arr_shift,
                                epsilon_shift,
                                name_list[i] + "_" + shift_name_list[s],
                                path=os.path.join(filepath, "shift/"),
                            )

                            df_shift.to_csv(
                                os.path.join(
                                    filepath,
                                    "shift/",
                                    f"grqa_measures_shift{rot_str}.csv",
                                ),
                                index=False,
                            )

                            shift_x += step_size
                            s += 1

                        shift_y += step_size

            df.to_csv(os.path.join(path, "grqa_measures.csv"), index=False)

            ### mass distribution
            
            #max_box_size_large = int(((np.min([ys_new, xs_new]) * 0.15) // box_size_store) + 1) * box_size_store
            #max_box_size = np.min([max_box_size, max_box_size_large])

            #mass_func_args = {
            #"nthreads": nthreads,
            #"max_box_size_large": max_box_size_large,
            #"box_size_store": box_size_store,
            #}

            (
                mass_distribution,
                mass_distribution_reverse,
                mass_distribution_z2,
                mass_distribution_reverse_z2,
                nr_boxes,
            ) = mass_func(
                arr,
                epsilon=epsilon,
                max_box_size=max_box_size,
                **mass_func_args,
            )

            dist_df = pd.DataFrame(
                {
                    "box_size": np.arange(max_box_size_large + 1),
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
            dist_df.to_csv(
                f"{filepath}/{df.loc[i]["name"]}_mass_distribution.csv", index=False
            )

        except:
            print("Something went wrong... Skip...")
            continue

    print("TOTAL TIME:", time.time() - t1)
