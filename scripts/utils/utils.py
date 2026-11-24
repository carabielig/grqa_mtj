import os

import cartopy.crs as ccrs
import cmasher as cmr
import geopandas as gpd
import largestinteriorrectangle as lir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import shapely
import tqdm
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
from rasterio.mask import mask as riomask
from scipy.ndimage import median_filter
from scipy.spatial.distance import pdist
from skimage.util import view_as_blocks

import scripts.utils.test_images as ti


def powerlaw(x, beta0, beta1):
    return beta0 * x**beta1


def get_differing_indices(i1, d1, l1, i2, d2, l2, l, y_setoff_i1=0):
    """
    Function to get differing indices between two approaches for a certain line length.
    https://stackoverflow.com/questions/71001188/how-to-find-a-missing-row-between-two-similar-2d-numpy-arrays

    y_setoff_i1: this is a helper if e.g. one tries to compare a square image to a rectangular image that has the
    same features but shifted in y direction. (E.g. for the test functions in tests/test_line_sampling.py)
    """
    i1_setoff = i1.copy()
    i1_setoff[:, [0, 2]] = i1_setoff[:, [0, 2]] + y_setoff_i1
    i1_stack = np.hstack([i1_setoff, d1, np.expand_dims(l1, axis=1)])
    i2_stack = np.hstack([i2, d2, np.expand_dims(l2, axis=1)])

    idx_missing_in_i1 = ~np.all(i2_stack == i1_stack[:, None], axis=2).any(axis=0)
    idx_missing_in_i2 = ~np.all(i1_stack == i2_stack[:, None], axis=2).any(axis=0)

    # this is for filtering out all features on the main diagonal HS
    idx_hs_i1 = (np.sum(d1[:, [0, 1]] == d1[:, [2, 3]], axis=1) == 2) & (
        np.sum(i1[:, [0, 1]] == i1[:, [2, 3]], axis=1) == 2
    )
    idx_hs_i2 = (np.sum(d2[:, [0, 1]] == d2[:, [2, 3]], axis=1) == 2) & (
        np.sum(i2[:, [0, 1]] == i2[:, [2, 3]], axis=1) == 2
    )

    missing_in_i1 = i2_stack[idx_missing_in_i1 & (l2 == l)]  # & ~idx_hs_i2]
    missing_in_i2 = i1_stack[idx_missing_in_i2 & (l1 == l)]  # & ~idx_hs_i1]

    return missing_in_i1, missing_in_i2


def add_large_ints(a, b, max_value=100):
    """
    Add two large integers whole sum would not fit in int64.
    """

    if a > max_value or b > max_value:
        print("a and/or b are larger than the allowed maximum value.")
        return 0, 0
    ueberlauf = 0
    rest = a
    if max_value - a < b:
        ueberlauf += 1
        rest -= max_value - b
    else:
        rest = a + b
    print(a, "+", b, "=", rest, "+", ueberlauf, "*", max_value)
    return ueberlauf, rest


def multiply_large_ints(a, b, max_value_pot=4):
    """
    Multiply two large integers whose product would not fit into int64.
    """

    pot = max_value_pot
    max_value = 10**pot  # max value = 10000 --> store numbers up until 9999
    sqrt_pot = int(
        (pot - 1) / 2
    )  # multiplying 2 values of this length does not get larger than max value

    # split the numbers at sqrt_pot
    a_upper = int(a / 10 ** (sqrt_pot + 1))
    a_lower = int(a - a_upper * 10 ** (sqrt_pot + 1))

    b_upper = int(b / 10 ** (sqrt_pot + 1))
    b_lower = int(b - b_upper * 10 ** (sqrt_pot + 1))

    # get the middle terms
    term1 = a_upper * b_lower
    term2 = a_lower * b_upper

    term1_upper = int(term1 / 10 ** (sqrt_pot + 1))
    term1_lower = int(term1 - term1_upper * 10 ** (sqrt_pot + 1))

    term2_upper = int(term2 / 10 ** (sqrt_pot + 1))
    term2_lower = int(term2 - term2_upper * 10 ** (sqrt_pot + 1))

    # get first results
    result_upper = a_upper * b_upper + term1_upper + term2_upper
    result_lower = a_lower * b_lower

    # check if the two term_lower produce overflow
    term_lower = term1_lower + term2_lower

    c = 0
    if max_value - result_lower <= term_lower * 10 ** (sqrt_pot + 1):
        c = 1
        result_lower = result_lower - (max_value - term_lower * 10 ** (sqrt_pot + 1))
    else:
        result_lower = result_lower + term_lower * 10 ** (sqrt_pot + 1)

    result_upper = result_upper + c

    return result_upper, result_lower


def estimate_epsilon(img, RR, mode="corner"):
    """
    Estimate the recurrence threshold epsilon. Three different modes are available:
    - "corner": use the top left corner (50 x 50 pixels) of the image to estimate epsilon
    - "random": tile the image in 50 x 50 px tiles and estimate epsilon from a random tile
    - "whole": estimate epsilon from the whole image (only possible if rp fits into memory)
    """

    print("Estimating recurrence threshold...")

    if (img.shape[0] >= 50) and (img.shape[1] >= 50) and (mode == "random"):
        blocks = view_as_blocks(
            img[: 50 * (img.shape[0] // 50), : 50 * (img.shape[1] // 50)],
            (img.shape[0] // 50, img.shape[1] // 50),
        )
        blocks = blocks.reshape(
            (blocks.shape[0] * blocks.shape[1], blocks.shape[2], blocks.shape[3])
        )

        new_img = np.zeros(blocks.shape[0])

        for i in range(blocks.shape[0]):
            new_img[i] = np.random.choice(blocks[i, :, :].flatten())

        epsilon = np.quantile(
            pdist(
                np.c_[new_img, new_img],
                metric="euclidean",
            ),
            0.2,
        )
    elif (img.shape[0] >= 50) and (img.shape[1] >= 50) and (mode == "corner"):
        ys, xs = img.shape
        est_px = np.min((ys, xs, 50))
        epsilon = np.quantile(
            pdist(
                np.c_[img[:est_px, :est_px].flatten(), img[:est_px, :est_px].flatten()],
                metric="euclidean",
            ),
            RR,
        )
    elif mode == "whole":
        epsilon = np.quantile(
            pdist(
                np.c_[img.flatten(), img.flatten()],
                metric="euclidean",
            ),
            RR,
        )
    else:
        epsilon = np.quantile(
            pdist(
                np.c_[img.flatten(), img.flatten()],
                metric="euclidean",
            ),
            RR,
        )

    return epsilon


def get_new_shape(ys, xs, max_block_nr=65536, nthreads=(32, 32), shape="square"):
    """
    Get shape that fits into the number of available GPU threads. Two modes are available:
    - "square": largest possible square within the image
    - else: scale the image (by keeping the aspect ratio) to the number of threads.
    """

    assert nthreads[0] == nthreads[1], "Number of threads must be equal."

    max_prod = (max_block_nr - 1) * nthreads[0]

    if shape == "square":
        if ys * xs >= max_prod:
            if np.min((ys, xs)) >= int(np.sqrt(max_prod)):
                ys_new = int(np.sqrt(max_prod))
                xs_new = int(np.sqrt(max_prod))
            else:
                a = (ys * xs) / max_prod
                ys_new = np.min((int(ys // np.sqrt(a)), int(xs // np.sqrt(a))))
                xs_new = ys_new
        else:
            ys_new = np.min((ys, xs))
            xs_new = np.min((ys, xs))

    else:
        if ys * xs >= max_prod:
            a = (ys * xs) / max_prod
            ys_new = int(ys // np.sqrt(a))
            xs_new = int(xs // np.sqrt(a))
        else:
            ys_new = ys
            xs_new = xs

    return ys_new, xs_new


def generate_test_images(size, path="data/test_images", save=False):
    """
    Generate (and save) the test patterns.
    """

    test_images = [
        ti.get_longitudinal_wave_test_image(size),
        ti.get_periodic_test_image(size),
        # np.pad(
        #    ti.get_fractal_test_image(int(np.emath.logn(3, size))),
        #    [
        #        (size - 3 ** int(np.emath.logn(3, size))) // 2,
        #        (size - 3 ** int(np.emath.logn(3, size))) // 2
        #        + (size - 3 ** int(np.emath.logn(3, size))) % 2,
        #    ],
        #    mode="maximum",
        # ),
        ti.get_fractal_test_image(int(np.emath.logn(3, size))),
        ti.get_turing_test_image(size),
        ti.get_logistic_map_test_image(size),
        ti.get_correlated_noise_test_image(size),
        ti.get_gaussian_distributed_white_noise(size),
        ti.get_uniformly_distributed_white_noise(size),
        ti.get_periodic_test_image(size, n=30) + ti.get_periodic_test_image(size, n=24),
        ti.get_periodic_test_image(size, n=10) + ti.get_periodic_test_image(size, n=15),
    ]

    epsilon_list = [None, None, 0, None, None, None, None, None, None, None]

    name_list = [
        "longitudinal",
        "periodic",
        "fractal",
        "turing",
        "logistic_map",
        "corr_noise",
        "gaussian_wn",
        "uniform_wn",
        "periodic_30_24",
        "periodic_10_15",
    ]

    real_name_list = [
        "Longitudinal Wave",
        "Periodic Pattern",
        "Fractal",
        "Turing Pattern",
        "Logistic Map",
        "Correlated Noise",
        "Gaussian White Noise",
        "Uniform. Dist. White Noise",
        "Periodic Pattern (30 24)",
        "Periodic Pattern(10 15)",
    ]

    if save:
        df = pd.DataFrame(
            {"name": name_list, "epsilon": epsilon_list, "real_name": real_name_list}
        )
        df.to_csv(os.path.join(path, "df.csv"), index=False)

        for i, img in enumerate(test_images):
            np.save(os.path.join(path, name_list[i] + ".npy"), img)
    else:
        return test_images, epsilon_list, name_list


def collect_test_images(filepath="data/test_images"):
    """
    Collect the test patterns from the respective data directory.
    """

    df = pd.read_csv(os.path.join(filepath, "df.csv"))

    name_list = list(df.name)
    epsilon_list = list(df.epsilon)

    test_images = [
        np.load(os.path.join(filepath, file + ".npy"), allow_pickle=True)
        for file in name_list
    ]

    return test_images, epsilon_list, name_list


def collect_bacteria_images(filepath="data/bacteria_images", smooth=False):
    """
    Collect the bacteria images from the respective data directory.
    """

    filelist = [
        f
        for f in os.listdir(filepath)
        if ((os.path.isfile(os.path.join(filepath, f))) and (f.endswith("png")))
    ]

    all_images = []
    name_list = []

    for n, f in enumerate(filelist):

        # im = plt.imread(os.path.join(filepath, f))

        # if f.startswith("sto"):

        #    im = im[10:-10, 10:-10]
        #    im[:, :, 1] *= 10
        #    im[:, :, 2] *= 20
        #    im = np.sum(im, axis=2)
        #    c = 0

        #    for i in np.unique(im):
        #        im[im == i] = c
        #        c += 1

        #    all_images.append(im)
        #    name_list.append(f.split(".")[0])

        # elif f.startswith("det"):

        #    im = im[30:-30, 30:-30]
        #    im_new = np.zeros(im.shape[:2])

        #    im_new[(im[:, :, 0] > 0.5) & (im[:, :, 1] < 0.3) & (im[:, :, 2] < 0.3)] = (
        #        2  # r
        #    )
        #    im_new[(im[:, :, 1] > 0.5) & (im[:, :, 0] < 0.5) & (im[:, :, 2] < 0.5)] = (
        #        3  # g
        #    )
        #    im_new[(im[:, :, 2] > 0.5) & (im[:, :, 0] < 0.5) & (im[:, :, 1] < 0.5)] = (
        #        1  # b
        #    )
        #    im_new[im_new == 0] = 2

        #    all_images.append(im_new)
        #    name_list.append(f.split(".")[0])

        # else:

        img = Image.open(os.path.join(filepath, f))
        # size = eval(f.split(".")[0])
        size = 512

        if size < 1024:
            img = img.crop((5, 5, 1029, 1029))
            img_small = img.resize((size, size), resample=Image.NEAREST)
            arr = np.array(img_small)
        else:
            arr = np.array(img)
            ys_new, xs_new = get_new_shape(
                arr.shape[0], arr.shape[1], nthreads=(16, 16)
            )
            arr = arr[
                arr.shape[0] // 2 - ys_new // 2 : arr.shape[0] // 2 + ys_new // 2,
                arr.shape[1] // 2 - xs_new // 2 : arr.shape[1] // 2 + xs_new // 2,
            ]

        _, indices = np.unique(
            arr.reshape(-1, arr.shape[2]), axis=0, return_inverse=True
        )
        im = indices.reshape(arr.shape[:2])

        if smooth:
            im = median_filter(im, size=10)

        all_images.append(im)
        name_list.append(f.split(".")[0])

    epsilon_list = [0] * len(name_list)

    return all_images, epsilon_list, name_list


def cut_and_save_dtm_and_slp(
    filepath_dtm="data/dtm",
    filepath_slp="data/slp",
    mask_filepath="data/mask",
    dtm_lir_filepath="data/dtm_lir",
    slp_lir_filepath="data/slp_lir",
    overwrite=False,
    start_with=0,
    nthreads=(16, 16),
):
    """
    Cut out the largest interior rectangle from the DTMs and slope arrays/river catchments and save.
    """

    filelist = [
        f
        for f in os.listdir(filepath_dtm)
        if (os.path.isfile(os.path.join(filepath_dtm, f)))
    ]

    for cm in tqdm.trange(start_with, len(filelist)):

        ### load the data

        cm_nr = filelist[cm].split("_")[0]

        if cm_nr in ["4358843", "4368009"]:
            f"{cm_nr} does not exist/is wrong. Skip..."
            continue

        if (
            (not overwrite)
            and (os.path.exists(os.path.join(slp_lir_filepath, f"{cm_nr}_lin_slp.tif")))
            and (os.path.exists(os.path.join(dtm_lir_filepath, f"{cm_nr}_lin.tif")))
        ):
            print(f"{cm_nr} exists already. Skip...")
            continue

        if (
            (not os.path.exists(os.path.join(filepath_dtm, f"{cm_nr}_lin.tif")))
            or (not os.path.exists(os.path.join(filepath_slp, f"{cm_nr}_lin_slp.tif")))
            or (not os.path.exists(os.path.join(mask_filepath, f"mask_{cm_nr}.tif")))
        ):
            print(f"WARNING: missing file for {cm_nr}. Skip...")
            continue

        dataset_dtm = rio.open(os.path.join(filepath_dtm, f"{cm_nr}_lin.tif"))
        dtm = dataset_dtm.read().squeeze()

        dataset_slp = rio.open(os.path.join(filepath_slp, f"{cm_nr}_lin_slp.tif"))
        slp = dataset_slp.read().squeeze()

        dataset_mask = rio.open(os.path.join(mask_filepath, f"mask_{cm_nr}.tif"))
        mask = dataset_mask.read().squeeze()

        ### find largest interior rectangle (lir)

        # https://github.com/OpenStitching/lir
        rec = lir.lir(mask.astype(bool))

        ### cut to lir extent

        mask[rec[1] : rec[1] + rec[3], rec[0] : rec[0] + rec[2]] = 2
        dtm[mask == 0] = np.nan
        slp[mask == 0] = np.nan

        ### get lir layer

        bbox_layer = np.ones_like(mask) * np.nan
        bbox_layer[rec[1] : rec[1] + rec[3], rec[0] - 50 : rec[0] + 50] = 1
        bbox_layer[
            rec[1] : rec[1] + rec[3], rec[0] + rec[2] - 50 : rec[0] + rec[2] + 50
        ] = 1
        bbox_layer[rec[1] - 50 : rec[1] + 50, rec[0] : rec[0] + rec[2]] = 1
        bbox_layer[
            rec[1] + rec[3] - 50 : rec[1] + rec[3] + 50, rec[0] : rec[0] + rec[2]
        ] = 1

        ### get GPU extent

        ys_new, xs_new = get_new_shape(rec[3], rec[2], nthreads=nthreads)

        bbox_layer_gpu = np.ones_like(mask) * np.nan
        bbox_layer_gpu[rec[1] : rec[1] + ys_new, rec[0] - 50 : rec[0] + 50] = 1
        bbox_layer_gpu[
            rec[1] : rec[1] + ys_new, rec[0] + xs_new - 50 : rec[0] + xs_new + 50
        ] = 1
        bbox_layer_gpu[rec[1] - 50 : rec[1] + 50, rec[0] : rec[0] + xs_new] = 1
        bbox_layer_gpu[
            rec[1] + ys_new - 50 : rec[1] + ys_new + 50, rec[0] : rec[0] + xs_new
        ] = 1

        ### plot dtm

        # https://stackoverflow.com/questions/75729096/add-cartopy-coastlines-to-subplots
        proj = ccrs.UTM(10)
        extent = np.array(dataset_dtm.bounds)[[0, 2, 1, 3]]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": proj})

        cmap = cmr.get_sub_cmap("terrain", 0.2, 0.75)

        im = ax.imshow(dtm, cmap=cmap, extent=extent, vmin=0, vmax=300)
        ax.imshow(bbox_layer, cmap="autumn", extent=extent)
        ax.imshow(bbox_layer_gpu, cmap="winter", extent=extent)
        ax.coastlines("10m")

        cbar = plt.colorbar(
            im, extend="max", orientation="horizontal", aspect=25, pad=0.08
        )
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label="elevation [m]", fontsize=12, labelpad=10)

        gl = ax.gridlines(
            color="black",
            alpha=0.4,
            linestyle="dotted",
            draw_labels=True,
            dms=False,
            x_inline=False,
            y_inline=False,
        )
        gl.top_labels = False
        gl.right_labels = False

        ax.add_artist(ScaleBar(1))

        handles, labels = plt.gca().get_legend_handles_labels()
        line1 = Line2D(
            [dtm.shape[0] // 2], [dtm.shape[1] // 2], label="coastline", color="k"
        )
        line2 = Line2D(
            [dtm.shape[0] // 2],
            [dtm.shape[1] // 2],
            label="largest interior\nrectangle (lir)",
            color="red",
            lw=3,
        )
        line3 = Line2D(
            [dtm.shape[0] // 2],
            [dtm.shape[1] // 2],
            label="lir scaled to\nnr of GPU blocks",
            color="blue",
            lw=3,
        )
        handles.extend([line1, line2, line3])

        ax.legend(handles=handles, fontsize=12, loc="lower right")

        ax.set_title(f"Catchment Nr. {cm_nr}", fontsize=20, fontweight="bold", pad=20)

        # plt.show()
        plt.savefig(f"img/dtm/{cm_nr}.png", dpi=300, bbox_inches="tight")
        plt.close()

        ### plot slp

        # https://stackoverflow.com/questions/75729096/add-cartopy-coastlines-to-subplots
        proj = ccrs.UTM(10)
        extent = np.array(dataset_dtm.bounds)[[0, 2, 1, 3]]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={"projection": proj})

        cmap = plt.get_cmap("BuPu")

        im = ax.imshow(slp, cmap=cmap, extent=extent, vmin=0, vmax=50)
        ax.imshow(bbox_layer, cmap="autumn", extent=extent)
        ax.imshow(bbox_layer_gpu, cmap="winter", extent=extent)
        ax.coastlines("10m")

        cbar = plt.colorbar(
            im, extend="max", orientation="horizontal", aspect=25, pad=0.08
        )
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label="slope [°]", fontsize=12, labelpad=10)

        gl = ax.gridlines(
            color="black",
            alpha=0.4,
            linestyle="dotted",
            draw_labels=True,
            dms=False,
            x_inline=False,
            y_inline=False,
        )
        gl.top_labels = False
        gl.right_labels = False

        ax.add_artist(ScaleBar(1))

        handles, labels = plt.gca().get_legend_handles_labels()
        line1 = Line2D(
            [dtm.shape[0] // 2], [dtm.shape[1] // 2], label="coastline", color="k"
        )
        line2 = Line2D(
            [dtm.shape[0] // 2],
            [dtm.shape[1] // 2],
            label="largest interior\nrectangle",
            color="red",
            lw=3,
        )
        line3 = Line2D(
            [dtm.shape[0] // 2],
            [dtm.shape[1] // 2],
            label="lir scaled to\nnr of GPU blocks",
            color="blue",
            lw=3,
        )
        handles.extend([line1, line2, line3])

        ax.legend(handles=handles, fontsize=12, loc="lower right")

        ax.set_title(f"Catchment Nr. {cm_nr}", fontsize=20, fontweight="bold", pad=20)

        # plt.show()
        plt.savefig(f"img/slp/{cm_nr}.png", dpi=300, bbox_inches="tight")
        plt.close()

        ### cut and save dtm

        bbox_xy = shapely.Polygon(
            [
                (rec[1], rec[0]),
                (rec[1], rec[0] + rec[2]),
                (rec[1] + rec[3], rec[0] + rec[2]),
                (rec[1] + rec[3], rec[0]),
                (rec[1], rec[0]),
            ]
        )
        bbox = shapely.Polygon(
            zip(*rio.transform.xy(dataset_dtm.transform, *bbox_xy.exterior.coords.xy))
        )

        out_image, out_transform = riomask(dataset_dtm, [bbox], crop=True)
        out_meta = dataset_dtm.meta

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rio.open(
            os.path.join(dtm_lir_filepath, f"{cm_nr}_lin.tif"), "w", **out_meta
        ) as dest:
            dest.write(out_image)

        ### cut and save slp

        out_image, out_transform = riomask(dataset_slp, [bbox], crop=True)
        out_meta = dataset_dtm.meta

        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )

        with rio.open(
            os.path.join(slp_lir_filepath, f"{cm_nr}_lin_slp.tif"), "w", **out_meta
        ) as dest:
            dest.write(out_image)


def collect_dtm(filepath_dtm="data/dtm_lir"):
    """
    Collect the DTMs from the respective data directory.
    """

    filelist = [
        f
        for f in os.listdir(filepath_dtm)
        if (os.path.isfile(os.path.join(filepath_dtm, f)))
    ]

    img_list = []
    name_list = []

    for cm in tqdm.tqdm(filelist):

        cm_nr = cm.split("_")[0]

        dataset_dtm = rio.open(os.path.join(filepath_dtm, f"{cm_nr}_lin.tif"))
        dtm = dataset_dtm.read().squeeze()[1:, 1:]

        img_list.append(dtm)
        name_list.append(cm_nr + "_dtm")

    epsilon_list = [None] * len(name_list)

    return img_list, epsilon_list, name_list


def collect_slp(filepath_slp="data/slp_lir"):
    """
    Collect the slope arrays from the respective data directory.
    """

    filelist = [
        f
        for f in os.listdir(filepath_slp)
        if (os.path.isfile(os.path.join(filepath_slp, f)))
    ]

    img_list = []
    name_list = []

    for cm in tqdm.tqdm(filelist):

        cm_nr = cm.split("_")[0]

        dataset_slp = rio.open(os.path.join(filepath_slp, f"{cm_nr}_lin_slp.tif"))
        slp = dataset_slp.read().squeeze()[1:, 1:]

        img_list.append(slp)
        name_list.append(cm_nr + "_slp")

    epsilon_list = [None] * len(name_list)

    return img_list, epsilon_list, name_list


def generate_white_noise(shapes, path="data/white_noise", overwrite=False, repeat=1):
    """
    Generate white noise images and save in the respective data directory.
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    for i in range(repeat):
        for s in shapes:
            if (
                os.path.exists(os.path.join(path, f"{s[0]}_{s[1]}.npy"))
                and not overwrite
            ):
                print("File exists already, continue...")
                continue
            img = np.random.uniform(size=np.prod(s)).reshape(s)
            np.save(os.path.join(path, f"{s[0]}_{s[1]}_{i}.npy"), img)


def collect_white_noise(filepath="data/white_noise"):
    """
    Collect the white noise images from the respective data directory.
    """

    filelist = [
        f
        for f in os.listdir(filepath)
        if ((os.path.isfile(os.path.join(filepath, f)) and (f.endswith(".npy"))))
    ]

    test_images = [
        np.load(os.path.join(filepath, file), allow_pickle=True) for file in filelist
    ]

    name_list = [f.split(".")[0] for f in filelist]

    return test_images, [np.nan] * len(test_images), name_list


def generate_checkerboard(path="data/checkerboard", overwrite=False):
    """
    Generate checkerboard images and save in the respective data directory.
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    img = ti.get_special_checkerboard(20)
    np.save(os.path.join(path, f"special.npy"), img)

    img = ti.get_checkerboard(1, 10)
    np.save(os.path.join(path, f"1px.npy"), img)

    img = ti.get_checkerboard(2, 5)
    np.save(os.path.join(path, f"2px.npy"), img)

    img = ti.get_checkerboard(5, 2)
    np.save(os.path.join(path, f"5px.npy"), img)


def collect_checkerboard(filepath="data/checkerboard"):

    filelist = [
        f
        for f in os.listdir(filepath)
        if ((os.path.isfile(os.path.join(filepath, f)) and (f.endswith(".npy"))))
    ]

    test_images = [
        np.load(os.path.join(filepath, file), allow_pickle=True) for file in filelist
    ]

    name_list = [f.split(".")[0] for f in filelist]

    return test_images, [0] * len(test_images), name_list
