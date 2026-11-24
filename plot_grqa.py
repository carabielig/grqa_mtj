import os
import sys
from posixpath import join
from warnings import filterwarnings

import cartopy.crs as ccrs
import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib import cm
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LightSource
import matplotlib.ticker as mticker
from scipy.signal import argrelextrema, find_peaks
from scipy.stats import kstest, linregress
from scipy.ndimage import gaussian_filter
from tqdm import trange
from skimage.transform import downscale_local_mean

from scripts import grqa, utils

filterwarnings("ignore")

# path to results from rqa
path = "img/test_images"

# minimum length for lines to be detected in the rp
minlen = 2

# for detection of extrema
peak_prominence = 0.4
# bacteria images: 0.3
# test images lac: 0.2
# test images: 0.4 (fractal: 0.01)
# slp/dtm: 0.05


# %% load and prepare data

images, _, name_list = utils.collect_test_images()
#images, _, name_list = utils.collect_bacteria_images()
#images, _, name_list = utils.collect_bacteria_images(smooth=True)
#images, _, name_list = utils.collect_dtm()
#images, _, name_list = utils.collect_slp()
#images, _, name_list = utils.collect_white_noise()
#images, _, name_list = utils.collect_checkerboard()

# load rqa measure data frame
df = grqa.open_grqa_measures(path, minlen=minlen)

# drop unused data
df = df.drop(df.loc[df.name == "periodic_10_15"].index)
df = df.drop(df.loc[df.name == "periodic_30_24"].index)
df = df.drop(df.loc[df.name == "stochastic_1"].index)
df = df.drop(df.loc[df.name == "stochastic_3"].index)
df = df.drop(df.loc[df.name == "stochastic_4"].index)
df = df.drop(df.loc[df.name == "stochastic_5"].index)
df = df.drop(df.loc[df.name == "deterministic_1"].index)
df = df.drop(df.loc[df.name == "deterministic_3"].index)
df = df.drop(df.loc[df.name == "deterministic_4"].index)
df = df.drop(df.loc[df.name == "deterministic_5"].index)

# sort by name column
df["name"] = df.name.astype(str)
df = df.sort_values("name")

# sort more
if "bacteria" in path:
    pass
elif "test_images" in path:
    df = df.reindex([0, 1, 2, 3, 5, 4, 6, 7])

# for river catchments: add uplift rate and get resolution
if ("slp" in path) or ("dtm" in path):

    # load uplift rate df
    cm_df = pd.read_csv(
        "data/hebungsraten.csv",
        header=None,
        names=["catchment", "rate"],
        sep=" ",
        skipinitialspace=True,
    )
    cm_df["name"] = list(cm_df.catchment.astype(str) + "_" + path.split("/")[-1][:3])

    # add uplift rate to grqa measure df
    for row in df.index:
        if cm_df[cm_df.name == df.name[row]].rate.values.size != 0:
            df.loc[row, "hr"] = cm_df[cm_df.name == df.name[row]].rate.values[0]

    # interpolate (nearest) uplift rate
    df.loc[df.index[0], "hr"] = 0.5
    df["hr"] = df.hr.interpolate(method="linear")

    # get resolution
    resolution = eval(path.split("_")[1][0])


# %% replot images

if ("dtm" in path) or ("slp" in path):

    extent_list = []

    for i in trange(len(images)):

        cm_nr = name_list[i].split("_")[0]
        datatype = path.split("/")[-1].split("_")[0]
        res = path.split("/")[-1].split("_")[1]
        filepath_slp = f"data/{datatype}"
        mask_filepath = "data/mask"
        cut_filepath = f"data/{datatype}_lir"

        if datatype == "slp":
            dataset_slp = rio.open(os.path.join(filepath_slp, f"{cm_nr}_lin_slp.tif"))
            slp = dataset_slp.read().squeeze()
            dataset_cut = rio.open(os.path.join(cut_filepath, f"{cm_nr}_lin_slp.tif"))
            cut = dataset_cut.read().squeeze()
            cmap = plt.get_cmap("BuPu")
            vmin = 0
            vmax = 50
            cbar_label = "slope [°]"
        elif datatype == "dtm":
            dataset_slp = rio.open(os.path.join(filepath_slp, f"{cm_nr}_lin.tif"))
            slp = dataset_slp.read().squeeze()
            dataset_slp = rio.open(os.path.join(cut_filepath, f"{cm_nr}_lin.tif"))
            cut = dataset_cut.read().squeeze()
            cmap = cmr.get_sub_cmap("terrain", 0.2, 0.75)
            vmin = 0
            vmax = 300
            cbar_label = "elevation [m]"

        dataset_mask = rio.open(os.path.join(mask_filepath, f"mask_{cm_nr}.tif"))
        mask = dataset_mask.read().squeeze()

        # slope array
        slp[mask == 0] = np.nan
        extent = np.array(dataset_slp.bounds)[[0, 2, 1, 3]]

        # lir
        lir = np.ones_like(cut) * np.nan
        lir[:, -100:] = 1
        lir[:, :100] = 1
        lir[:100, :] = 1
        lir[-100:, :] = 1
        extent_lir = np.array(dataset_cut.bounds)[[0, 2, 1, 3]]
        extent_list.append(extent_lir)

        # gpu rectangle with resolution
        gpu = np.ones_like(cut) * np.nan
        ys, xs = gpu.shape
        ys_new, xs_new = utils.get_new_shape(ys, xs, nthreads=(16, 16))
        if (ys_new * resolution <= ys) and (xs_new * resolution <= xs):
            ys_new *= resolution
            xs_new *= resolution
        else:
            ys_new = np.min((ys, xs))
            xs_new = np.min((ys, xs))

        gpu = gpu[
            gpu.shape[0] // 2 - ys_new // 2 : gpu.shape[0] // 2 + ys_new // 2,
            gpu.shape[1] // 2 - xs_new // 2 : gpu.shape[1] // 2 + xs_new // 2,
        ]
        gpu[:, -100:] = 1
        gpu[:, :100] = 1
        gpu[:100, :] = 1
        gpu[-100:, :] = 1
        extent_gpu = extent_lir.copy()

        # same aspect ratio as lir
        # extent_gpu[1] = extent_gpu[0] + ((extent_gpu[1] - extent_gpu[0]) * (ys_new/ys))
        # extent_gpu[2] = extent_gpu[3] - ((extent_gpu[3] - extent_gpu[2]) * (xs_new/xs))

        # square in the center of a catchment
        extent_gpu[0] = extent_lir[0] + (extent_lir[1] - extent_lir[0]) / 2 - xs_new / 2
        extent_gpu[1] = extent_lir[0] + (extent_lir[1] - extent_lir[0]) / 2 + xs_new / 2
        extent_gpu[2] = extent_lir[3] - (extent_lir[3] - extent_lir[2]) / 2 - ys_new / 2
        extent_gpu[3] = extent_lir[3] - (extent_lir[3] - extent_lir[2]) / 2 + ys_new / 2

        ### plot catchment map

        # https://stackoverflow.com/questions/75729096/add-cartopy-coastlines-to-subplots
        proj = ccrs.UTM(10)

        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(8, 5), subplot_kw={"projection": proj}, gridspec_kw={'width_ratios': [3, 1]})

        # special cases to make maps prettier
        if cm_nr == "4490572":
            ax.set_xlim(extent[0], extent[1]-12500)
            ax.set_ylim(extent[2]+2500, extent[3]-7500)
        elif cm_nr == "4389142":
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2]-2000, extent[3]+2000)
        else:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
        ax.set_autoscale_on(False)

        im = ax.imshow(slp, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax, rasterized=True)

        # plot all rectangles to show variance
        step_size = 1000  # not necessary to scale to resolution, bc image is not downsampled here
        shift_y = 0
        res = np.ones((ys_new, xs_new, 1))
        while shift_y + ys_new <= cut.shape[0]:
            shift_x = 0
            while shift_x + xs_new <= cut.shape[1]:
                x_center = ((cut.shape[1] - xs_new) % step_size) // 2
                y_center = ((cut.shape[0] - ys_new) % step_size) // 2
                cut_shift = cut[
                    int(shift_y + y_center) : int(shift_y + ys_new + y_center),
                    int(shift_x + x_center) : int(shift_x + xs_new + x_center),
                ]
                res = np.concatenate([res, np.expand_dims(cut_shift, 2)], axis=2)
                extent_shift = extent_lir.copy()
                extent_shift[0] = extent_lir[0] + shift_x + x_center
                extent_shift[1] = extent_lir[0] + shift_x + xs_new + x_center
                extent_shift[2] = extent_lir[2] + shift_y + y_center
                extent_shift[3] = extent_lir[2] + shift_y + ys_new + y_center
                ax.imshow(gpu, cmap="Pastel2_r", extent=extent_shift)
                shift_x += step_size
            shift_y += step_size

        # plot lir and qpu rectangle
        ax.imshow(lir, cmap="autumn", extent=extent_lir)
        ax.imshow(gpu, cmap="winter", extent=extent_gpu)
        ax.coastlines("10m")

        cbar = plt.colorbar(
            im, extend="max", orientation="horizontal", aspect=25, pad=0.15
        )
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(label=cbar_label, fontsize=16, labelpad=10)

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
            [slp.shape[0] // 2], [slp.shape[1] // 2], label="coastline", color="k"
        )
        line2 = Line2D(
            [slp.shape[0] // 2],
            [slp.shape[1] // 2],
            label="largest interior\nrectangle",
            color="red",
            lw=3,
        )
        line3 = Line2D(
            [slp.shape[0] // 2],
            [slp.shape[1] // 2],
            label="center square\nscaled to\nnr of GPU threads",
            color="blue",
            lw=3,
        )
        line4 = Line2D(
            [slp.shape[0] // 2],
            [slp.shape[1] // 2],
            label=f"variation of the\ncenter square\nin {step_size}m steps",
            color="silver",
            lw=3,
        )
        handles.extend([line1, line2, line3, line4])

        ax1.legend(handles=handles, loc="center", fontsize=12, framealpha=1,
                        edgecolor="k",
                        borderpad=0.6)
        ax1.set_axis_off()
        fig.suptitle(f"Catchment Nr. {cm_nr} (Map)", fontsize=20, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # plt.show()
        plt.savefig(os.path.join(path, f"{cm_nr}.png"), dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(path, f"{cm_nr}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()

        ### slp_lir center square only

        fig, ax = plt.subplots(1, 1, figsize=(5, 7))#, subplot_kw={"projection": proj})

        cmap = plt.get_cmap("BuPu")

        if resolution != 1:
            downscale = downscale_local_mean(
                cut[
                    cut.shape[0] // 2 - ys_new // 2 : cut.shape[0] // 2 + ys_new // 2,
                    cut.shape[1] // 2 - xs_new // 2 : cut.shape[1] // 2 + xs_new // 2,
                ],
                (resolution, resolution),
            )
        else:
            downscale = cut[
                cut.shape[0] // 2 - ys_new // 2 : cut.shape[0] // 2 + ys_new // 2,
                cut.shape[1] // 2 - xs_new // 2 : cut.shape[1] // 2 + xs_new // 2,
            ]

        im = ax.imshow(downscale, cmap=cmap, vmin=0, vmax=50, rasterized=True)#, extent=extent_gpu)

        cbar = plt.colorbar(
            im, extend="max", orientation="horizontal", aspect=25, pad=0.08
        )
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(label="slope [°]", fontsize=16, labelpad=10)

        ax.set_aspect('equal')
        tick_spacing = 50
        ax.xaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(tick_spacing))

        ax.set_xlabel("x [px]", fontsize=16, labelpad=15)
        ax.set_ylabel("y [px]", fontsize=16, labelpad=15)
        ax.tick_params(axis='x', labelsize=14, top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.tick_params(axis='y', labelsize=14)
        ax.xaxis.set_label_position('top')

        #gl = ax.gridlines(
        #    color="black",
        #    alpha=1,
        #    linestyle="dotted",
        #    draw_labels=True,
        #    dms=False,
        #    x_inline=False,
        #    y_inline=False,
        #    # crs=proj # not supported :(
        #)
        #gl.top_labels = False
        #gl.right_labels = False

        #ax.add_artist(ScaleBar(1, box_alpha=0.7))

        ax.set_title(f"Catchment Nr. {cm_nr}\nCenter Square\n(Slope)", fontsize=20, fontweight="bold", pad=70)

        # plt.show()
        try:
            plt.savefig(
                os.path.join(path, name_list[i], f"{name_list[i]}_map.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(path, name_list[i], f"{name_list[i]}_map.pdf"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        except FileNotFoundError:
            continue

        ### DTM and hillshade for specific catchments

        #if (cm_nr == "4371394") or (cm_nr == "4490572"):
        try:
            with rio.open(os.path.join("data", "dtm_lir", f"{cm_nr}_lin.tif")) as dataset_dtm_cut:
                dtm = dataset_dtm_cut.read().squeeze()

            cmap = cmr.get_sub_cmap("gist_earth", 0.35, 0.8)

            vmin = 0
            vmax = 350

            dtm = dtm[
                        cut.shape[0] // 2 - ys_new // 2: cut.shape[0] // 2 + ys_new // 2,
                        cut.shape[1] // 2 - xs_new // 2: cut.shape[1] // 2 + xs_new // 2,
                        ]

            ls = LightSource(azdeg=315, altdeg=45)
            shaded = ls.shade(dtm, cmap=cmap, vmin=vmin, vmax=vmax,
                              vert_exag=15, blend_mode='soft')

            fig, ax = plt.subplots(1, 1, figsize=(5, 7))  # , subplot_kw={"projection": proj})
            im = ax.imshow(shaded, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)

            ax.set_xticks(np.arange(0, dtm.shape[1], resolution*50), np.arange(0, dtm.shape[1] // resolution, 50))
            ax.set_yticks(np.arange(0, dtm.shape[0], resolution * 50),
                          np.arange(0, dtm.shape[0] // resolution, 50))

            cbar = plt.colorbar(
                im, extend="max", orientation="horizontal", aspect=25, pad=0.08
            )
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label(label="elevation [m]", fontsize=16, labelpad=10)

            ax.set_xlabel("x [px]", fontsize=16, labelpad=15)
            ax.set_ylabel("y [px]", fontsize=16, labelpad=15)
            ax.tick_params(axis='x', labelsize=14, top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.tick_params(axis='y', labelsize=14)
            ax.xaxis.set_label_position('top')

            # gl = ax.gridlines(
            #    color="black",
            #    alpha=1,
            #    linestyle="dotted",
            #    draw_labels=True,
            #    dms=False,
            #    x_inline=False,
            #    y_inline=False,
            #    # crs=proj # not supported :(
            # )
            # gl.top_labels = False
            # gl.right_labels = False

            #ax.add_artist(ScaleBar(1, box_alpha=0.7))

            ax.set_title(f"Catchment Nr. {cm_nr}\nCenter Square\n(DTM and Hillshade)", fontsize=20, fontweight="bold", pad=70)

            # plt.show()
            plt.savefig(
                os.path.join(path, name_list[i], f"{name_list[i]}_hillshade.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(path, name_list[i], f"{name_list[i]}_hillshade.pdf"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        except:
            pass


elif ("test_images" in path) or ("white" in path) or ("checkerboard" in path):

    pattern_df = pd.read_csv(os.path.join("data", path.split("/")[1], "df.csv"), sep=",")

    for i in pattern_df.index:

        fig, ax = plt.subplots(1, 1, figsize=(5, 6))

        #map = ax.imshow(images[i], cmap="Greys_r", origin="lower", rasterized=True)
        map = ax.imshow(images[i], cmap="Blues_r", origin="lower", rasterized=True)
        ax.set_title(pattern_df.real_name[i], fontsize=20, fontweight="bold", pad=20)
        ax.set_xlabel("x", fontsize=16, labelpad=10)
        ax.set_ylabel("y", fontsize=16, labelpad=10)
        ax.tick_params(axis='both', labelsize=14)

        if ("checkerboard" in path):
            ax.set_xticks([0, 5, 10, 15])
            ax.set_yticks([0, 5, 10, 15])

        cbar = plt.colorbar(map, fraction=0.046, ax=ax)
        cbar.set_label("pixel value", fontsize=16, labelpad=10)
        cbar.ax.tick_params(labelsize=14)

        plt.savefig(
            os.path.join(path, pattern_df.name[i] + "_blue.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(path, pattern_df.name[i] + "_blue.pdf"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

elif "bacteria" in path:

    pattern_df = pd.read_csv(os.path.join("data", "bacteria_images", "df.csv"), sep=",")

    for i in range(len(images)):

        fig, ax = plt.subplots(1, 1, figsize=(5, 6))

        cmap = cm.get_cmap("tab20c_r", 4)

        img = images[i].copy()
        if not ("det" in name_list[i]):
            img[img == 3] = 4
            img[img == 0] = 3
            img[img == 4] = 0

        map = ax.imshow(img, cmap=cmap, origin="lower", vmin=-0.5, vmax=3.5, rasterized=True)
        ax.set_title(pattern_df.real_name[i], fontsize=20, fontweight="bold", pad=20)
        ax.set_xlabel("x", fontsize=16, labelpad=10)
        ax.set_ylabel("y", fontsize=16, labelpad=10)
        ax.tick_params(axis='both', labelsize=14)

        cbar = plt.colorbar(map, fraction=0.046, ax=ax)
        cbar.set_label("pixel value", fontsize=16, labelpad=10)
        cbar.ax.tick_params(labelsize=14)
        plt.savefig(
            os.path.join(path, name_list[i] + ".png"), dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(path, name_list[i] + ".pdf"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        plt.close("all")
        dpi = 100
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        plt.imshow(img, cmap=cmap, origin="lower", vmin=-0.5, vmax=3.5)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(os.path.join(path, name_list[i] + "_original.png"), dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close()

plt.close("all")
#np.save("data/extent_list.npy", np.array(extent_list))


#%% replot RPs


for i in trange(len(images)):

    try:
        epsilon = df[df.name == name_list[i]].epsilon.values[0]
    except IndexError:
        continue

    if "slp" in path:

        arr = images[i].copy()

        if resolution > 1:
            arr = downscale_local_mean(arr, (resolution, resolution))

        orig = arr.copy()

        ys_new, xs_new = utils.get_new_shape(
            arr.shape[0], arr.shape[1], nthreads=(16, 16)
        )

        arr = arr[
              arr.shape[0] // 2 - ys_new // 2: arr.shape[0] // 2 + ys_new // 2,
              arr.shape[1] // 2 - xs_new // 2: arr.shape[1] // 2 + xs_new // 2,
              ]

        utils.plot_rp_3d_new(arr, epsilon, plot_path=os.path.join(path, name_list[i]), name=f"{name_list[i]}_3drp", origin="upper")
    else:
        utils.plot_rp_3d_new(images[i], epsilon, plot_path=os.path.join(path, name_list[i]), name=f"{name_list[i]}_3drp")
        utils.plot_rp_3d_rev(images[i], epsilon, plot_path=os.path.join(path, name_list[i]), name=f"{name_list[i]}_3drp")


# %% mtj map, all catchments

if ("dtm" in path) or ("slp" in path):
    el = np.load("data/extent_list.npy")

    proj = ccrs.UTM(10)

    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(6, 8),
        subplot_kw={"projection": proj},
        gridspec_kw={"width_ratios": [3, 1]},
    )

    ax.set_xlim(np.min(el[:, 0]) - 10000, np.max(el[:, 1] + 10000))
    ax.set_ylim(np.min(el[:, 2]) - 10000, np.max(el[:, 3] + 10000))
    ax.set_autoscale_on(False)

    ax2.set_xlim(np.min(el[:, 0]), np.min(el[:, 0]) + 25000)
    ax2.set_ylim(np.min(el[:, 2]), np.max(el[:, 3]))
    ax2.set_autoscale_on(False)
    df["catchment_nr"] = [eval(n.split("_")[0]) for n in df.name]
    y_sort = df.catchment_nr.to_numpy()
    text_y = np.linspace(np.min(el[:, 2]), np.max(el[:, 3]), 47 + 6)
    text_y = np.delete(text_y, [1, 2, 6, 9, 47, 41])

    ax.coastlines("10m")

    # landsat background
    utils.load_and_plot_landsat_rgb(ax, prefix="LC08_L2SP_046033_20250805_20250815_02_T1_SR")
    utils.load_and_plot_landsat_rgb(ax, prefix="LC08_L2SP_045033_20250814_20250821_02_T1_SR")
    utils.load_and_plot_landsat_rgb(ax, prefix="LC08_L2SP_045032_20250814_20250821_02_T1_SR")
    utils.load_and_plot_landsat_rgb(ax, prefix="LC08_L2SP_046032_20250805_20250815_02_T1_SR")

    cmap = plt.get_cmap("gist_heat_r")
    norm = Normalize(vmin=0, vmax=5)

    for i, ex in enumerate(el):
        hr = df.hr[df.name == name_list[i]].values[0]
        img = np.ones((10, 10)) * 5
        img[1:-1, 1:-1] = hr
        #im = ax.imshow(img, cmap=cmap, extent=ex, vmin=0, vmax=5, interpolation="None", rasterized=True)
        rect = Rectangle(
            (ex[0], ex[2]),  # Bottom-left corner
            ex[1] - ex[0],
            ex[3] - ex[2],
            facecolor=cmap(hr/5),
            edgecolor="black",
            linewidth=0.5,
            transform=proj,
        )
        ax.add_patch(rect)
        text_loc = np.where(y_sort == eval(name_list[i].split("_")[0]))[0][0]
        ax2.text(
            np.min(el[:, 0]),
            text_y[text_loc],
            name_list[i].split("_")[0],
            size=9,
            verticalalignment="center",
        )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="uplift rate [mm/yr]")

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
    ax.set_title(
        f"Coastal River Catchments\nat the MTJ", fontsize=20, fontweight="bold", pad=30
    )

    ax2.set_title("Catchment\nNumber", pad=20, loc="left")
    ax2.set_axis_off()

    plt.subplots_adjust(wspace=0.05)

    #plt.show()
    plt.savefig("img/latex/mtj_map.png", dpi=300, bbox_inches="tight")
    plt.savefig("img/latex/mtj_map.pdf", dpi=300, bbox_inches="tight")
    plt.close()


# %% plot distributions (line length/patch size/mass)

bs_intercept = 10

for i in trange(len(images)):

    ### load line length distribution (rot) + histogram

    try:
        ll_dist_df = pd.read_csv(
            os.path.join(
                path, name_list[i], f"{name_list[i]}_line_length_distribution_rot.csv"
            )
        )
    except FileNotFoundError:
        continue

    if minlen == 2:
        minlen_str = ""
    else:
        minlen_str = f"_{minlen}"

    diag_dir_fac = 64
    vert_dir_fac = 8
    utils.plot_hist(
        ll_dist_df.diag_frequ[minlen-2:],
        ll_dist_df.length[minlen-2:],
        diag_dir_fac,
        ll_dist_df.vert_frequ[minlen-2:],
        ll_dist_df.length[minlen-2:],
        vert_dir_fac,
        suptitle=f"Line Sampling (with rotation) - Minimum Length = {minlen}",
        filepath=os.path.join(
            path, name_list[i], f"line_length_distribution_rot_hist{minlen_str}.png"
        ),
    )

    if "smooth" in path:

        diag_dir_fac = 64
        vert_dir_fac = 8

        utils.plot_hist(
            ll_dist_df.diag_frequ[minlen - 2:99],
            ll_dist_df.length[minlen - 2:99],
            diag_dir_fac,
            ll_dist_df.vert_frequ[minlen - 2:99],
            ll_dist_df.length[minlen - 2:99],
            vert_dir_fac,
            suptitle=f"Line Sampling (with rotation) - Min. L. = {minlen}, Max. L. = 100",
            filepath=os.path.join(
                path, name_list[i], f"line_length_distribution_rot_hist{minlen_str}_100.png"
            ),
        )

        lengths = pd.read_csv(f"img/bacteria_images_smooth/{name_list[i]}/Results.csv", index_col=0).Length
        lengths = np.round(lengths).astype(int)
        lengths_summary1 = pd.DataFrame(
            {"Mean [px]": int(np.round(lengths.mean())), "Std. [px]": int(np.ceil(lengths.std()))},
            index=[0])
        lengths_summary2 = pd.DataFrame(
            {"Mean/3 [px]": int(np.round(lengths.mean() / 3)), "Std./3 [px]": int(np.ceil(lengths.std() / 3))},
            index=[0])

        fig, ax = plt.subplots(1, 2, figsize=(2, 2), gridspec_kw={"width_ratios": [1, 2]})

        ax[0].axis('off')
        ax[0].axis('tight')
        ax[0].table(cellText=np.expand_dims(lengths.to_numpy(), 1), colLabels=["Lengths [px]"], loc="center")

        ax[1].axis('off')
        ax[1].axis('tight')
        ax[1].table(cellText=lengths_summary1.values, colLabels=lengths_summary1.columns, loc="upper center")
        ax[1].table(cellText=lengths_summary2.values, colLabels=lengths_summary2.columns, loc="lower center")

        fig.tight_layout()

        #plt.show()
        plt.savefig(f"img/bacteria_images_smooth/{name_list[i]}/lengths_table.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"img/bacteria_images_smooth/{name_list[i]}/lengths_table.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    ### load patch size distribution (rot) + histogram

    try:
        ps_dist_df = pd.read_csv(
            os.path.join(
                path + "_patch", name_list[i], f"{name_list[i]}_line_length_distribution_rot.csv"
            )
        )
        diag_dir_fac = 16
        vert_dir_fac = 4
        utils.plot_hist(
            ps_dist_df.diag_frequ[minlen-2:],
            ps_dist_df.length[minlen-2:],
            diag_dir_fac,
            ps_dist_df.vert_frequ[minlen-2:],
            ps_dist_df.length[minlen-2:],
            vert_dir_fac,
            suptitle=f"Patch Sampling (with rotation) - Minimum Length = {minlen}",
            filepath=os.path.join(
                path, name_list[i], f"patch_size_distribution_rot_hist{minlen_str}.png"
            ),
        )
    except FileNotFoundError:
        pass


    ### load mass distribution and lacunarity

    try:
        mass_dist_df = pd.read_csv(
            os.path.join(path, name_list[i], f"{name_list[i]}_mass_distribution.csv")
        )
    except FileNotFoundError:
        continue

    lac = mass_dist_df.lacunarity
    lac[lac == 0] = np.nan
    bs = mass_dist_df.box_size

    ### load not normalized lacunarity

    #lac_normal = grqa.lacunarity_not_normalized(mass_dist_df.mass_distribution.to_numpy(),
    #                                            mass_dist_df.mass_distribution_z2.to_numpy(),
    #                                            mass_dist_df.nr_boxes.to_numpy())
    #mass_dist_df["lac_normal"] = lac_normal

    ### mass distribution histogram

    if path == "img/test_images":
        mass_dist_df.loc[100, ["mass_distribution", "mass_distribution_reverse", "mass_distribution_z2",
                               "mass_distribution_reverse_z2"]] = mass_dist_df.loc[100, ["mass_distribution",
                                                                                         "mass_distribution_reverse",
                                                                                         "mass_distribution_z2",
                                                                                         "mass_distribution_reverse_z2"]] / 2
    try:
        utils.plot_hist_mass(
            bs,
            mass_dist_df.mass_distribution,
            mass_dist_df.mass_distribution_z2,
            suptitle="Sum of (Squared) Masses",
            filepath=os.path.join(
                path, name_list[i], f"mass_distribution_hist.png"
            ),
        )
    except:
        #print("No mass histogram bc sum of masses is too large...")
        pass

    #pattern_df = pd.read_csv(os.path.join("data", path.split("/")[1], "df.csv"), sep=",")

    # comparison to longitudinal wave pattern
    #utils.plot_hist_mass(
    #    bs,
    #    mass_dist_df.mass_distribution,
    #    mass_dist_df.mass_distribution_z2,
    #    suptitle="Sum of (Squared) Masses - Comparison",
    #    filepath=os.path.join(
    #        path, name_list[i], f"mass_distribution_hist_with_long.png"
    #    ),
    #    add_longitudinal=True,
    #    data_label=pattern_df[pattern_df.name==name_list[i]].real_name.values[0]
    #)

    ### lacunarity

    idx = (~np.isnan(lac)) & (lac > 0) & (bs <= images[i].shape[0]/2)
    bs = bs[idx].to_numpy()
    lac = lac[idx].to_numpy()

    # idx_max = argrelextrema(lac, np.greater)[0]
    # idx_min = argrelextrema(lac, np.less)[0]

    idx_max, _ = find_peaks(np.log10(lac), prominence=peak_prominence)
    idx_min, _ = find_peaks(-np.log10(lac), prominence=peak_prominence)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.loglog(bs, lac, c="darkmagenta", lw=2)

    if bs_intercept == 0:
        bs_intercept = np.max(bs) * 0.1  # at 10%

    if bs_intercept in bs:
        intercept = lac[bs == bs_intercept]
    else:
        intercept = lac[(np.abs(bs - bs_intercept)).argmin()]

    try:
        idx = df[df.name == name_list[i]].index[0]
        df.loc[idx, f"intercept_{int(bs_intercept)}"] = intercept
    except IndexError:
        pass

    # for detection of extrema

    #if idx_min.size != 0:
    #    ax.scatter(
    #        bs[idx_min],
    #        lac[idx_min],
    #        label=f"Minima at:\nw = " + ", ".join(f"{v}" for v in bs[idx_min]),
    #        c="forestgreen",
    #        alpha=0.5,
    #        s=200
    #    )

    #if idx_max.size != 0:
    #    ax.scatter(
    #        bs[idx_max],
    #        lac[idx_max],
    #        c="navy",
    #        alpha=0.5,
    #        s=200,
    #        label=f"Maxima at:\nw = " + ", ".join(f"{v}" for v in bs[idx_max]),
    #    )

    ax.set_ylabel("(normalized) lacunarity", fontsize=18)
    ax.set_xlabel("box size (w)", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid()
    ax.set_axisbelow(True)
    #ax.set_title(f"{pattern_df[pattern_df.name==name_list[i]].real_name.values[0]}", fontsize=18, fontweight="bold", pad=10)
    ax.set_title(f"{name_list[i]}", fontsize=22, fontweight="bold", pad=10)

    #ax.legend(loc="lower left", fontsize=14, framealpha=1,
    #                    edgecolor="k",
    #                    borderpad=0.6)

    plt.savefig(
        os.path.join(path, name_list[i], "lacunarity.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(path, name_list[i], "lacunarity.pdf"), dpi=300, bbox_inches="tight"
    )
    # plt.show()
    plt.close()

    ### not normalized lacunarity

    #bs = mass_dist_df.box_size
    #idx = (~np.isnan(lac_normal)) & (lac_normal > 0) & (bs <= images[i].shape[0]/2)
    #bs = bs[idx].to_numpy()
    #lac_normal = lac_normal[idx]

    #fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    #ax.loglog(bs, lac_normal, c="darkmagenta", lw=2, label="LAC")

    #ax.set_ylabel("Lacunarity (not normalized)", fontsize=14)
    #ax.set_xlabel("Box size (w)", fontsize=14)
    #ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.grid()
    #ax.set_axisbelow(True)
    #ax.set_title(f"{pattern_df[pattern_df.name==name_list[i]].real_name.values[0]}", fontsize=18, fontweight="bold", pad=10)

    ##ax.legend()

    #plt.savefig(
    #    os.path.join(path, name_list[i], "lacunarity_not_normalized.png"), dpi=300, bbox_inches="tight"
    #)
    ## plt.show()
    #plt.close()


#%% plot slope distributions

if "slp" in path:
    for i in trange(len(images)):
        img = images[i]
        be = np.arange(0, 100, 5)

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

        counts, bins = np.histogram(img.flatten(), bins=be)
        rel_freq = counts / counts.sum()

        ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge', edgecolor='black',
               facecolor="lightskyblue")

        percentile_90 = np.percentile(img.flatten(), 90)

        idx = df[df.name == name_list[i]].index[0]
        df.loc[idx, f"percentile_90"] = percentile_90

        mean = np.mean(img.flatten())
        std = np.std(img.flatten())
        df.loc[idx, f"slope_std"] = std

        ax.axvline(mean, lw=2, c="k", linestyle="--", label=f"mean: {mean:.1f}°")
        ax.axvspan(mean - std, mean + std, color="steelblue", alpha=0.5, lw=0, label=f"std: {std:.1f}°")
        ax.axvline(percentile_90, lw=2, c="k", linestyle="-.", label=f"90th perc.:\n{percentile_90:.1f}°")

        ax.set_xlabel('slope [°]', fontsize=14, labelpad=10)
        ax.set_ylabel('relative frequency', fontsize=14, labelpad=10)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_ylim(0, 0.2)
        ax.tick_params(axis='both', labelsize=12)

        plt.legend(loc="upper right", fontsize=12, framealpha=1,
                        edgecolor="k",
                        borderpad=0.6)

        ax.grid()
        ax.set_axisbelow(True)
        ax.set_title(f"Catchment nr. {name_list[i].split("_")[0]}", fontsize=16, pad=20)
        fig.suptitle("Histogram of Slope Values", fontsize=18, fontweight="bold")
        fig.subplots_adjust(top=0.73)

        #plt.show()
        plt.savefig(
           os.path.join(path, name_list[i], "slope_distribution.png"), dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(path, name_list[i], "slope_distribution.pdf"), dpi=300, bbox_inches="tight"
        )
        plt.close()


# %% grqa measure summary

for add_line in [True, False]:

    if ("slp" in path):
        fig, ax = plt.subplots(3, 2, figsize=(12, 14))
    else:
        fig, ax = plt.subplots(3, 2, figsize=(12, 14))

    min_arr = np.ones((5, len(df))) * np.nan
    max_arr = np.ones((5, len(df))) * np.nan
    if ("slp" in path):

        for nr, n in enumerate(df.name):

            try:
                shift_df = pd.read_csv(os.path.join(path, n, "shift", "grqa_measures_shift_rot.csv"))
            except FileNotFoundError:
                continue

            min_arr[:, nr] = shift_df[["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot", "entropy_line_rot"]].min().to_numpy()
            max_arr[:, nr] = shift_df[["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot", "entropy_line_rot"]].max().to_numpy()

        min_arr = np.abs(np.moveaxis(df[["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot",
                                         "entropy_line_rot"]].to_numpy(), 0, 1) - min_arr)
        max_arr = np.abs(np.moveaxis(df[["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot",
                             "entropy_line_rot"]].to_numpy(), 0, 1) - max_arr)

    #ax[0, 0].scatter(np.arange(len(df)), df.det_line, label="det_line", c="steelblue", s=100, alpha=0.5)
    df.det_line_rot[df.det_line_rot > 1] = 1
    ax[0, 0].errorbar(
        np.arange(len(df)), df.det_line_rot, yerr=np.vstack([min_arr[0, :], max_arr[0, :]]),
        marker="o", lw = 0, elinewidth=2, label="det_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[0, 0].grid()
    ax[0, 0].set_title("Determinism (DET)", fontweight="bold", pad=10)
    ax[0, 0].set_axisbelow(True)
    ax[0, 0].set_ylabel(r"[$\cdot$100%]", fontsize=12, labelpad=10)

    #ax[0, 1].scatter(
    #    np.arange(len(df)), df.l_hs_line, label="l_hs_line", c="steelblue", s=100, alpha=0.5
    #)
    ax[0, 1].errorbar(
        np.arange(len(df)), df.l_hs_line_rot, yerr=np.vstack([min_arr[1, :], max_arr[1, :]]),
        marker="o", lw = 0, elinewidth=2, label="l_hs_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[0, 1].grid()
    ax[0, 1].set_title(
        r"Average diag. line length/patch size (L)", fontweight="bold", pad=10
    )
    ax[0, 1].set_axisbelow(True)
    ax[0, 1].set_ylabel(r"[px]", fontsize=12, labelpad=10)

    #ax[2, 0].scatter(np.arange(len(df)), df.lam_line, label="lam_line", c="steelblue", s=100, alpha=0.5)
    ax[2, 0].errorbar(
        np.arange(len(df)), df.lam_line_rot, yerr=np.vstack([min_arr[2, :], max_arr[2, :]]),
        marker="o", lw = 0, elinewidth=2, label="lam_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[2, 0].grid()
    ax[2, 0].set_title("Laminarity (LAM)", fontweight="bold", pad=10)
    ax[2, 0].set_axisbelow(True)
    ax[2, 0].set_ylabel(r"[$\cdot$100%]", fontsize=12, labelpad=10)

    #ax[2, 1].scatter(np.arange(len(df)), df.tt_line, c="steelblue", s=100, alpha=0.5)
    ax[2, 1].errorbar(
        np.arange(len(df)), df.tt_line_rot, yerr=np.vstack([min_arr[3, :], max_arr[3, :]]),
        marker="o", lw = 0, elinewidth=2, c="darkblue", markersize=10, alpha=0.5
    )
    ax[2, 1].grid()
    ax[2, 1].set_title("Trapping Size (TS)", fontweight="bold", pad=10)
    ax[2, 1].set_axisbelow(True)
    ax[2, 1].set_ylabel(r"[px]", fontsize=12, labelpad=10)

    #ax[1, 0].scatter(
    #    np.arange(len(df)), df.entropy_line, label="ent_line", c="steelblue", s=100, alpha=0.5
    #)
    ax[1, 0].errorbar(
        np.arange(len(df)), df.entropy_line_rot, yerr=np.vstack([min_arr[4, :], max_arr[4, :]]),
        marker="o", lw = 0, elinewidth=2, label="ent_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[1, 0].grid()
    ax[1, 0].set_title("Entropy (ENT)", fontweight="bold", pad=10)
    ax[1, 0].set_axisbelow(True)

    add_line_string = ""
    if add_line:
        ax[0, 0].plot(np.arange(len(df)), df.det_line_rot, c="darkblue", zorder=2)
        ax[0, 1].plot(np.arange(len(df)), df.l_hs_line_rot, c="darkblue", zorder=2)
        ax[2, 0].plot(np.arange(len(df)), df.lam_line_rot, c="darkblue", zorder=2)
        ax[2, 1].plot(np.arange(len(df)), df.tt_line_rot, c="darkblue", zorder=2)
        ax[1, 0].plot(np.arange(len(df)), df.entropy_line_rot, c="darkblue", zorder=2)
        add_line_string = "_line"

    ### add patch sampling

    #path_patch = "img/test_images_patch"
    #df_patch = grqa.open_grqa_measures(path_patch, minlen=minlen)
    #df_patch = df_patch.sort_values("name")
    #df_patch = df_patch.reindex([0, 1, 2, 3, 5, 4, 6, 7])
#
    #ax[0, 0].scatter(np.arange(len(df_patch)), df_patch.det_line, label="det_patch", c="lightcoral", s=100, alpha=0.5)
    #ax[0, 0].scatter(
    #    np.arange(len(df_patch)), df_patch.det_line_rot, label="det_patch_rot", c="crimson", s=100, alpha=0.5
    #)
#
    #ax[0, 1].scatter(
    #    np.arange(len(df_patch)), df_patch.l_hs_line, label="l_hs_patch", c="lightcoral", s=100, alpha=0.5
    #)
    #ax[0, 1].scatter(
    #    np.arange(len(df_patch)), df_patch.l_hs_line_rot, label="l_hs_patch_rot", c="crimson", s=100, alpha=0.5
    #)
#
    #ax[2, 0].scatter(np.arange(len(df_patch)), df_patch.lam_line, label="lam_patch", c="lightcoral", s=100, alpha=0.5)
    #ax[2, 0].scatter(
    #    np.arange(len(df_patch)), df_patch.lam_line_rot, label="lam_patch_rot", c="crimson", s=100, alpha=0.5
    #)
#
    #ax[2, 1].scatter(np.arange(len(df_patch)), df_patch.tt_line, c="lightcoral", s=100, alpha=0.5)
    #ax[2, 1].scatter(
    #    np.arange(len(df_patch)), df_patch.tt_line_rot, c="crimson", s=100, alpha=0.5
    #)
#
    #ax[1, 0].scatter(
    #    np.arange(len(df_patch)), df_patch.entropy_line, label="ent_patch", c="lightcoral", s=100, alpha=0.5
    #)
    #ax[1, 0].scatter(
    #    np.arange(len(df_patch)), df_patch.entropy_line_rot, label="ent_patch_rot", c="crimson", s=100, alpha=0.5
    #)
    #
    #if add_line:
    #    ax[0, 0].plot(np.arange(len(df_patch)), df_patch.det_line_rot, c="crimson", zorder=2)
    #    ax[0, 1].plot(np.arange(len(df_patch)), df_patch.l_hs_line_rot, c="crimson", zorder=2)
    #    ax[2, 0].plot(np.arange(len(df_patch)), df_patch.lam_line_rot, c="crimson", zorder=2)
    #    ax[2, 1].plot(np.arange(len(df_patch)), df_patch.tt_line_rot, c="crimson", zorder=2)
    #    ax[1, 0].plot(np.arange(len(df_patch)), df_patch.entropy_line_rot, c="crimson", zorder=2)

    # https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend
    handles, labels = plt.gca().get_legend_handles_labels()

    point1 = Line2D(
        [0],
        [0],
        label="line sampling",
        marker="o",
        markersize=10,
        alpha=0.5,
        markerfacecolor="steelblue",
        markeredgecolor="steelblue",
        linestyle="",
    )

    point2 = Line2D(
        [0],
        [0],
        label="line sampling (with rotation)",
        marker="o",
        markersize=10,
        alpha=0.5,
        markerfacecolor="darkblue",
        markeredgecolor="darkblue",
        linestyle="",
    )

    line2 = Line2D(
        [0],
        [0],
        label="variability obtained from\nshifted squares",
        color="darkblue",
        linestyle="-",
    )

    point3 = Line2D(
        [0],
        [0],
        label="patch sampling",
        marker="o",
        markersize=10,
        alpha=0.5,
        markerfacecolor="lightcoral",
        markeredgecolor="lightcoral",
        linestyle="",
    )

    point4 = Line2D(
        [0],
        [0],
        label="patch sampling (with rotation)",
        marker="o",
        markersize=10,
        alpha=0.5,
        markerfacecolor="crimson",
        markeredgecolor="crimson",
        linestyle="",
    )

    handels = []
    if ("slp" in path) or ("dtm" in path):
        line = Line2D([0], [0], label="uplift rate [mm/yr]", color="grey", lw=2)
        handles.extend([point2, line2, line])
    else:
        #handles.extend([point1, point2, point3, point4])
        handles.extend([point2])

    ax[1, 1].legend(handles=handles, loc="center", fontsize=14, framealpha=1,
            edgecolor="k",
            borderpad=0.6)
    ax[1, 1].set_axis_off()

    if ("smooth" in path) or ("bacteria" in path):
        ax[0, 0].set_ylim(-0.1, 1.1)
        ax[2, 0].set_ylim(-0.1, 1.1)
        ax[1, 0].set_ylim(-0.1, 5)
        ax[0, 1].set_ylim(1, 35)
        ax[2, 1].set_ylim(1, 35)
    elif ("checkerboard" in path):
        ax[0, 0].set_ylim(-0.1, 1.1)
        ax[2, 0].set_ylim(-0.1, 1.1)
        ax[1, 0].set_ylim(-0.1, 2)
        ax[0, 1].set_ylim(0, 12)
        ax[2, 1].set_ylim(0, 12)
    elif ("test_images" in path):
        ax[0, 0].set_ylim(-0.1, 1.1)
        ax[2, 0].set_ylim(-0.1, 1.1)
        ax[1, 0].set_ylim(-0.1, 2)
        ax[0, 1].set_ylim(1, 6.5)
        ax[2, 1].set_ylim(1, 6.5)
    elif ("slp" in path):
        ax[0, 0].set_ylim(-0.1, 1.1)
        ax[2, 0].set_ylim(-0.1, 1.1)
        ax[1, 0].set_ylim(-0.1, 2)
        ax[0, 1].set_ylim(1, 6.5)
        ax[2, 1].set_ylim(1, 6.5)

    for a in ax.flatten()[[0, 1, 2, 4, 5]]:
        if ("slp" in path) or ("dtm" in path):
            ax_tw = a.twinx()
            ax_tw.plot(
                np.arange(len(df)), df.hr, c="grey", lw=2, label="uplift rate", zorder=1
            )
            ax_tw.set_ylabel(f"uplift rate [mm/yr]", fontsize=12)
            a.set_xticks(np.arange(len(df)), labels=[n.split("_")[0] for n in df.name], rotation=90, fontsize=10)
            a.set_xlabel("catchment nr.", fontsize=12)
        elif ("test_images" in path) or ("white" in path):
            pattern_df = pd.read_csv(
                os.path.join("data", "test_images", "df.csv"), sep=","
            )
            df2 = df.merge(pattern_df, on="name")
            a.set_xticks(np.arange(len(df)), labels=df2.real_name, rotation=90, fontsize=12)
        elif ("checkerboard" in path):
            pattern_df = pd.read_csv(
                os.path.join("data", "checkerboard", "df.csv"), sep=","
            )
            df2 = df.merge(pattern_df, on="name")
            a.set_xticks(np.arange(len(df)), labels=df2.short_name, rotation=90, fontsize=12)
        elif "bacteria" in path:
            pattern_df = pd.read_csv(
                os.path.join("data", "bacteria_images", "df.csv"), sep=","
            )
            pattern_df["name"] = pattern_df["name"].astype(str)
            df2 = df.merge(pattern_df, on="name")
            a.set_xticks(np.arange(len(df)), labels=df2.real_name, rotation=90, fontsize=12)
        a.yaxis.set_tick_params(labelsize=12)

    if ("slp" in path) or ("dtm" in path):
        plt.subplots_adjust(hspace=0.6, wspace=0.2)
    else:
        plt.subplots_adjust(hspace=0.8, wspace=0.4)

    #plt.show()
    plt.savefig(
        os.path.join(path, f"grqa_measures{add_line_string}_minlen_{minlen}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(path, f"grqa_measures{add_line_string}_minlen_{minlen}.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# %% grqa measure summary (2 figures for 2 pages)

add_line_string=""
if "slp" in path:

    min_arr = np.ones((5, len(df))) * np.nan
    max_arr = np.ones((5, len(df))) * np.nan

    for nr, n in enumerate(df.name):

        try:
            shift_df = pd.read_csv(os.path.join(path, n, "shift", "grqa_measures_shift_rot.csv"))
        except FileNotFoundError:
            continue

        min_arr[:, nr] = shift_df[
            ["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot", "entropy_line_rot"]].min().to_numpy()
        max_arr[:, nr] = shift_df[
            ["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot", "entropy_line_rot"]].max().to_numpy()

    min_arr = np.abs(np.moveaxis(df[["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot",
                                     "entropy_line_rot"]].to_numpy(), 0, 1) - min_arr)
    max_arr = np.abs(np.moveaxis(df[["det_line_rot", "l_hs_line_rot", "lam_line_rot", "tt_line_rot",
                                     "entropy_line_rot"]].to_numpy(), 0, 1) - max_arr)

    fig, ax = plt.subplots(3, 1, figsize=(12, 14))

    df.det_line_rot[df.det_line_rot > 1] = 1
    ax[0].errorbar(
        np.arange(len(df)), df.det_line_rot, yerr=np.vstack([min_arr[0, :], max_arr[0, :]]),
        marker="o", lw = 0, elinewidth=2, label="det_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[0].grid()
    ax[0].set_title("Determinism (DET)", fontweight="bold", pad=10, fontsize=18)
    ax[0].set_axisbelow(True)
    ax[0].set_ylabel(r"[$\cdot$100%]", fontsize=16, labelpad=10)

    ax[1].errorbar(
        np.arange(len(df)), df.l_hs_line_rot, yerr=np.vstack([min_arr[1, :], max_arr[1, :]]),
        marker="o", lw = 0, elinewidth=2, label="l_hs_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[1].grid()
    ax[1].set_title(
        r"Average diag. line length/patch size (L)", fontweight="bold", pad=10, fontsize=18
    )
    ax[1].set_axisbelow(True)
    ax[1].set_ylabel(r"[px]", fontsize=16, labelpad=10)

    ax[2].errorbar(
        np.arange(len(df)), df.entropy_line_rot, yerr=np.vstack([min_arr[4, :], max_arr[4, :]]),
        marker="o", lw = 0, elinewidth=2, label="ent_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[2].grid()
    ax[2].set_title("Entropy (ENT)", fontweight="bold", pad=10, fontsize=18)
    ax[2].set_axisbelow(True)

    if ("smooth" in path) or ("bacteria" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(1, 6.5)
        ax[2].set_ylim(-0.1, 2)
    elif ("checkerboard" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(0, 12)
        ax[2].set_ylim(-0.1, 2)
    elif ("test_images" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(1, 6.5)
        ax[2].set_ylim(-0.1, 2)
    elif ("slp" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(1, 6.5)
        ax[2].set_ylim(-0.1, 2)

    for a in ax.flatten():
        ax_tw = a.twinx()
        ax_tw.plot(
            np.arange(len(df)), df.hr, c="grey", lw=2, label="uplift rate", zorder=1
        )
        ax_tw.set_ylabel(f"uplift rate [mm/yr]", fontsize=16, labelpad=10)
        a.set_xticks(np.arange(len(df)), labels=[n.split("_")[0] for n in df.name], rotation=90, fontsize=12)
        a.set_xlabel("catchment nr.", fontsize=16, labelpad=10)
        a.yaxis.set_tick_params(labelsize=14)
        ax_tw.yaxis.set_tick_params(labelsize=14)

    plt.subplots_adjust(hspace=0.8)

    # plt.show()
    plt.savefig(
        os.path.join(path, f"grqa_measures{add_line_string}_minlen_{minlen}_pt1.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(path, f"grqa_measures{add_line_string}_minlen_{minlen}_pt1.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


    fig, ax = plt.subplots(3, 1, figsize=(12, 14))

    ax[0].errorbar(
        np.arange(len(df)), df.lam_line_rot, yerr=np.vstack([min_arr[2, :], max_arr[2, :]]),
        marker="o", lw=0, elinewidth=2, label="lam_line_rot", c="darkblue", markersize=10, alpha=0.5
    )
    ax[0].grid()
    ax[0].set_title("Laminarity (LAM)", fontweight="bold", pad=10, fontsize=18)
    ax[0].set_axisbelow(True)
    ax[0].set_ylabel(r"[$\cdot$100%]", fontsize=16, labelpad=10)

    ax[1].errorbar(
        np.arange(len(df)), df.tt_line_rot, yerr=np.vstack([min_arr[3, :], max_arr[3, :]]),
        marker="o", lw=0, elinewidth=2, c="darkblue", markersize=10, alpha=0.5
    )
    ax[1].grid()
    ax[1].set_title("Trapping Size (TS)", fontweight="bold", pad=10, fontsize=18)
    ax[1].set_axisbelow(True)
    ax[1].set_ylabel(r"[px]", fontsize=16, labelpad=10)

    # https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend
    handles, labels = plt.gca().get_legend_handles_labels()

    point2 = Line2D(
        [0],
        [0],
        label="line sampling (with rotation)",
        marker="o",
        markersize=10,
        alpha=0.5,
        markerfacecolor="darkblue",
        markeredgecolor="darkblue",
        linestyle="",
    )

    line2 = Line2D(
        [0],
        [0],
        label="variability obtained from\nshifted squares",
        color="darkblue",
        linestyle="-",
    )

    point3 = Line2D(
        [0],
        [0],
        label="patch sampling",
        marker="o",
        markersize=10,
        alpha=0.5,
        markerfacecolor="lightcoral",
        markeredgecolor="lightcoral",
        linestyle="",
    )

    point4 = Line2D(
        [0],
        [0],
        label="patch sampling (with rotation)",
        marker="o",
        markersize=10,
        alpha=0.5,
        markerfacecolor="crimson",
        markeredgecolor="crimson",
        linestyle="",
    )

    handels = []
    line = Line2D([0], [0], label="uplift rate [mm/yr]", color="grey", lw=2)
    handles.extend([point2, line2, line])

    ax[2].legend(handles=handles, loc="center", fontsize=14, framealpha=1,
            edgecolor="k",
            borderpad=0.6)
    ax[2].set_axis_off()

    if ("smooth" in path) or ("bacteria" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(1, 6.5)
    elif ("checkerboard" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(0, 12)
    elif ("test_images" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(1, 6.5)
    elif ("slp" in path):
        ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(1, 6.5)

    for a in ax.flatten()[:-1]:
        ax_tw = a.twinx()
        ax_tw.plot(
            np.arange(len(df)), df.hr, c="grey", lw=2, label="uplift rate", zorder=1
        )
        ax_tw.set_ylabel(f"uplift rate [mm/yr]", fontsize=16, labelpad=10)
        a.set_xticks(np.arange(len(df)), labels=[n.split("_")[0] for n in df.name], rotation=90, fontsize=12)
        a.set_xlabel("catchment nr.", fontsize=16, labelpad=10)
        a.yaxis.set_tick_params(labelsize=14)
        ax_tw.yaxis.set_tick_params(labelsize=14)

    plt.subplots_adjust(hspace=0.8)

    # plt.show()
    plt.savefig(
        os.path.join(path, f"grqa_measures{add_line_string}_minlen_{minlen}_pt2.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(path, f"grqa_measures{add_line_string}_minlen_{minlen}_pt2.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


# %% grqa measures related to uplift rate

if ("slp" in path) or ("dtm" in path):

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.scatter(df.hr, df.det_line_rot, c="royalblue", label="DET")
    ax.scatter(df.hr, df.lam_line_rot, c="crimson", label="LAM")
    ax.scatter(df.hr, df.entropy_line_rot, c="olivedrab", label="ENT")

    ax.set_ylabel("GRQA Measure Value")
    ax.set_xlabel("Tectonic Uplift [mm/yr]")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_title(
        f"GRQA Measures - Tectonic Uplift", fontsize=14, fontweight="bold", pad=10
    )

    ax.legend(loc="center")

    plt.savefig(
        os.path.join(path, f"grqa_measures_hr_minlen_{minlen}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# %% all lacunarity curves

if ("white" in path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
else:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for i in range(len(images)):

    ### mass distribution

    try:
        mass_dist_df = pd.read_csv(
            os.path.join(path, name_list[i], f"{name_list[i]}_mass_distribution.csv")
        )
    except FileNotFoundError:
        continue

    if (name_list[i] == "periodic_30_24") or (name_list[i] == "periodic_10_15"):
        continue

    if (name_list[i].startswith("stochastic")) and (not name_list[i].endswith("2")):
        continue

    if (name_list[i].startswith("deterministic")) and (not name_list[i].endswith("2")):
        continue

    lac = mass_dist_df.lacunarity
    lac[lac == 0] = np.nan
    # lac = lac.fillna(method="ffill")
    bs = mass_dist_df.box_size
    max_bs = np.max(bs)
    #if name_list[i] == "4422784_slp":
    #    max_bs = ((images[i].shape[0] // resolution) * 0.15 // 10) * 10
    #    print(max_bs)
    #elif ("test" in path):
    #    max_bs = 100
    #else:
    #    max_bs = int(((np.min(images[i].shape) * 1/resolution * 0.15) // 10) + 1) * 10
    idx = (~np.isnan(lac)) & (lac > 0) & (bs <= max_bs)
    lac = lac[idx].to_numpy()
    bs = bs[idx].to_numpy()

    if ("slp" in path) or ("dtm" in path):
        color_list = ["gold", "tomato", "limegreen", "maroon"]
        if name_list[i] in list(df.name):
            cmap = plt.get_cmap("gist_heat_r")
            hr = df.hr[df.name == name_list[i]].values[0]
            ax.loglog(bs, lac, label=name_list[i], c=cmap(hr / 5), alpha=0.5)
            #part = df.part[df.name == name_list[i]].values[0]
            #ax.loglog(bs, lac, label=name_list[i], c=color_list[int(part)-1], alpha=0.5)
            #ax.loglog(bs, lac, c="silver", alpha=0.5)
        ax.set_xlim(0.8, 200)
        ax.set_ylim(0.5e-2, 2e0)
    elif ("test_images" in path) or ("checkerboard" in path):
        try:
            pattern_df = pd.read_csv(os.path.join("data", path.split("/")[1], "df.csv"), sep=",")
        except:
            pattern_df = pd.read_csv(os.path.join("data", "test_images", "df.csv"), sep=",")
        cmap = plt.get_cmap("turbo")
        ax.loglog(
            bs,
            lac,
            label=pattern_df.real_name[i],
            c=cmap(i / (len(name_list) - 1)),
            alpha=0.7,
            lw=2,
        )
    elif ("white" in path):

        pattern_df = pd.read_csv(os.path.join("data", path.split("/")[1], "df.csv"), sep=",")
        cmap = plt.get_cmap("turbo")

        if name_list[i].startswith("500"):
            ax[0].loglog(
                bs,
                lac,
                label=pattern_df.real_name[i],
                c=cmap(i / (len(name_list))),
                alpha=0.7,
                lw=2,
            )
        elif name_list[i].startswith("750"):
            ax[1].loglog(
                bs,
                lac,
                label=pattern_df.real_name[i],
                c=cmap(i / (len(name_list))),
                alpha=0.7,
                lw=2,
            )
        else:
            ax[2].loglog(
                bs,
                lac,
                label=pattern_df.real_name[i],
                c=cmap(i / (len(name_list))),
                alpha=0.7,
                lw=2,
            )


    elif "bacteria" in path:
        pattern_df = pd.read_csv(
            os.path.join("data", "bacteria_images", "df.csv"), sep=","
        )
        cmap = plt.get_cmap("turbo")

        ax.loglog(
            bs,
            lac,
            label=pattern_df.real_name[i],
            c=cmap(i / (len(name_list))),
            alpha=0.7,
            lw=2,
        )
    else:
        ax.loglog(bs, lac, label=name_list[i], c="indigo", alpha=0.5)

#cm1 = "4371394"
#cm2 = "4490572"
#try:
#   #mass_dist_df = pd.read_csv(
#   #    os.path.join(path, "periodic", f"periodic_mass_distribution.csv")
#   #)
#   mass_dist_df = pd.read_csv(
#       os.path.join(path, f"{cm1}_slp", f"{cm1}_slp_mass_distribution.csv")
#   )
#   lac = mass_dist_df.lacunarity
#   lac[lac == 0] = np.nan
#   bs = mass_dist_df.box_size
#   bs = bs[(~np.isnan(lac)) & (lac > 0)].to_numpy()
#   lac = lac[(~np.isnan(lac)) & (lac > 0)].to_numpy()
#   ax.loglog(bs, lac, c="dodgerblue", alpha=0.5, lw=5, label=f"catchment nr. {cm1}")
#except FileNotFoundError:
#   pass
#
#try:
#   #mass_dist_df = pd.read_csv(
#   #    os.path.join(path,"uniform_wn", f"uniform_wn_mass_distribution.csv")
#   #)
#   mass_dist_df = pd.read_csv(
#       os.path.join(path, f"{cm2}_slp", f"{cm2}_slp_mass_distribution.csv")
#   )
#   lac = mass_dist_df.lacunarity
#   lac[lac == 0] = np.nan
#   bs = mass_dist_df.box_size
#   bs = bs[(~np.isnan(lac)) & (lac > 0)].to_numpy()
#   lac = lac[(~np.isnan(lac)) & (lac > 0)].to_numpy()
#   ax.loglog(bs, lac, c="deeppink", alpha=0.5, lw=5, label=f"catchment nr. {cm2}")
#except FileNotFoundError:
#   pass

#ax.legend(loc="lower left", framealpha=1, edgecolor="k", borderpad=0.6)

#colname = [col for col in df.columns if col.startswith("intercept")][0]
#ax.axvline(eval(colname.split("_")[-1]), c="crimson", lw=3, linestyle="--")
#ax.axvline(10, c="crimson", lw=3, linestyle="--")

if isinstance(ax, np.ndarray):
    for a in ax:
        a.set_ylabel("lacunarity")
        a.set_xlabel("box size (w)")
        a.set_xlim(0, 150)
        a.grid()
        a.set_axisbelow(True)
        a.legend(framealpha=1, edgecolor="k", borderpad=0.6)
        fig.suptitle(f"Lacunarity Curves", fontsize=14, fontweight="bold")
else:
    ax.set_ylabel("lacunarity")
    ax.set_xlabel("box size (w)")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_title(f"Lacunarity Curves", fontsize=14, fontweight="bold", pad=10)

    if ("slp" in path) or ("dtm" in path):
        cmappable = cm.ScalarMappable(Normalize(0, 5))
        cmappable.set_cmap(cmap)
        cbar = plt.colorbar(cmappable, fraction=0.046, ax=ax)
        cbar.set_label("uplift rate [mm/yr]", labelpad=10)

    if ("bacteria" in path) or ("test_images" in path) or ("checkerboard" in path):
        ax.legend(framealpha=1, edgecolor="k", borderpad=0.6)

plt.savefig(os.path.join(path, "lac_comparison.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(path, "lac_comparison.pdf"), dpi=300, bbox_inches="tight")
# plt.show()
plt.close()


# %% all lacunarity minima/maxima

# https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array

for xtr in ["Minima", "Maxima"]:

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [2, 1]}
    )

    for i in range(len(df)):

        ### mass distribution

        try:
            mass_dist_df = pd.read_csv(
                os.path.join(
                    path,
                    df.name[df.index[i]],
                    f"{df.name[df.index[i]]}_mass_distribution.csv",
                )
            )
        except FileNotFoundError:
            continue

        lac = mass_dist_df.lacunarity
        lac[lac == 0] = np.nan
        bs = mass_dist_df.box_size
        bs = bs[(~np.isnan(lac)) & (lac > 0)].to_numpy()
        lac = lac[(~np.isnan(lac)) & (lac > 0)].to_numpy()

        ### plot

        # local minima
        if xtr == "Maxima":
            # idx_min = argrelextrema(lac, np.greater)[0]
            idx_min, _ = find_peaks(np.log10(lac), prominence=peak_prominence)
        else:
            # idx_min = argrelextrema(lac, np.less)[0]
            idx_min, _ = find_peaks(-np.log10(lac), prominence=peak_prominence)

        if idx_min.size != 0:
            ax.scatter(
                np.ones_like(idx_min) + i,
                bs[idx_min],
                c="indigo",
                alpha=0.7,
                s=150,
                label=f"{xtr.lower()}",
            )
            axh.bar(
                np.ones_like(idx_min) + i,
                idx_min.size,
                facecolor="indigo",
                edgecolor="indigo",
                alpha=0.7,
            )

    # ax.set_ylabel(f"{xtr}")
    ax.set_ylabel(f"Box size (w) of {xtr}", fontsize=12)
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_title(f"Lacunarity {xtr}", fontsize=14, fontweight="bold", pad=10)

    if ("slp" in path) or ("dtm" in path):
        ax.set_xlim(0, 24)
        ax.set_ylim(80, 300)
        ax.set_yticks(np.arange(80, 301, 20))
        # ax.set_yticks([1, 2], ["UR < 3", r"UR ≥ 3"])
        ax.set_xticks(np.arange(1, len(df) + 1), "", rotation=90)
        axh.set_xlim(0, 24)
    else:
        ax.set_xlim(0, len(df) + 1)
        ax.set_ylim(0, np.max(bs) + 10)
        ax.set_xticks(np.arange(1, len(df) + 1), "", rotation=90)
        axh.set_xlim(0, len(df) + 1)

    axh.grid()
    axh.set_axisbelow(True)
    axh.set_ylabel(f"Abs. Frequ.", fontsize=12)

    if ("slp" in path) or ("dtm" in path):
        axh.set_xticks(np.arange(1, len(df) + 1), df.name, rotation=90)
    elif "test_images" in path:
        pattern_df = pd.read_csv(os.path.join("data", "test_images", "df.csv"), sep=",")
        df2 = df.merge(pattern_df, on="name")
        axh.set_xticks(np.arange(1, len(df) + 1), labels=df2.real_name, rotation=90)
    elif "bacteria" in path:
        pattern_df = pd.read_csv(
            os.path.join("data", "bacteria_images", "df.csv"), sep=","
        )
        df2 = df.merge(pattern_df, on="name")
        axh.set_xticks(np.arange(1, len(df) + 1), labels=df2.real_name, rotation=90)
    else:
        axh.set_xticks(np.arange(1, len(df) + 1), df.name, rotation=90)

    if ("slp" in path) or ("dtm" in path):
        ax_tw = ax.twinx()
        ax_tw.plot(np.arange(len(df)), df.hr, c="dimgrey", lw=3, label="uplift rate")
        ax_tw.set_ylabel(f"uplift rate [mm/yr]", fontsize=12)

    try:
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_tw.get_legend_handles_labels()
        ax.legend(
            [h1[0]] + h2,
            [l1[0]] + l2,
            loc="lower right",
            framealpha=1,
            edgecolor="k",
            borderpad=0.6,
        )
    except:
        pass

    plt.subplots_adjust(hspace=0.1)

    # plt.show()
    plt.savefig(
        os.path.join(path, f"lac_{xtr.lower()}_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# %% lac intercept at a specific box size

add_line = False

# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
if ("slp" in path):
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
else:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax = np.array([ax])

colname = [col for col in df.columns if col.startswith("intercept")][0]
# colname = "intercept_30"
ax[0].scatter(
    np.arange(len(df)), df[colname], label=f"lac. at w = {colname.split("_")[-1]}", c="firebrick", alpha=0.7, s=100
)
ax[0].grid()
ax[0].set_ylabel(f"lacunarity", fontsize=12)
ax[0].set_title(f"Lacunarity at w = {colname.split("_")[-1]}", fontweight="bold", pad=10, fontsize=14)
ax[0].set_axisbelow(True)
ax[0].set_ylim(-0.1, 1)

if ("slp" in path) or ("dtm" in path):
    ax_tw = ax[0].twinx()
    ax_tw.plot(np.arange(len(df)), df.hr, c="dimgrey", lw=3, label="uplift rate")
    ax_tw.set_ylabel(f"uplift rate [mm/yr]", fontsize=12)
    ax_tw.set_ylim(0, 5)
    h2, l2 = ax_tw.get_legend_handles_labels()
    ax[0].set_xlabel("catchment nr.", fontsize=12)
else:
    h2, l2 = [], []

# ax[0].scatter(np.arange(len(df)), df.beta1, label="k", c="tomato", s=25)
# ax[0].grid()
# ax[0].set_ylabel("power law exponent", fontsize=12)
# ax[0].set_title("Power Law Exponent", fontweight="bold", pad=10)
# ax[0].set_axisbelow(True)

# ax[1].scatter(np.arange(len(df)), df.ks, label="KS", c="tomato")
# ax[1].grid()
# ax[1].set_title(
#    r"KS Statistic", fontweight="bold", pad=10
# )
# ax[1].set_axisbelow(True)

add_line_string = ""
if add_line:
   ax[0].plot(np.arange(len(df)), df.beta1, c="firebrick")
   #ax[1].plot(np.arange(len(df)), df.ks, c="tomato")
   add_line_string = "_line"

if "test_images" in path:
    pattern_df = pd.read_csv(os.path.join("data", "test_images", "df.csv"), sep=",")
    df2 = df.merge(pattern_df, on="name")
    for a in ax.flatten():
        a.set_xticks(np.arange(len(df)), labels=df2.real_name, rotation=90)
elif "bacteria" in path:
    pattern_df = pd.read_csv(os.path.join("data", "bacteria_images", "df.csv"), sep=",")
    df2 = df.merge(pattern_df, on="name")
    for a in ax.flatten():
        a.set_xticks(np.arange(len(df)), labels=df2.real_name, rotation=90)
elif "slp" in path:
    for a in ax.flatten():
        a.set_xticks(np.arange(len(df)), labels=[s.split("_")[0] for s in df.name], rotation=90)
else:
    for a in ax.flatten():
        a.set_xticks(np.arange(len(df)), labels=df.name, rotation=90)

h1, l1 = ax[0].get_legend_handles_labels()
if ("slp" in path):
    ax[0].legend(
        h1 + h2, l1 + l2, loc="upper left", framealpha=1, edgecolor="k", borderpad=0.6
    )
else:
    ax[0].legend(
        h1 + h2, l1 + l2, loc="upper right", framealpha=1, edgecolor="k", borderpad=0.6
    )

plt.subplots_adjust(hspace=0.5)

#plt.show()
plt.savefig(
    os.path.join(path, f"lac_measures{add_line_string}.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.savefig(
    os.path.join(path, f"lac_measures{add_line_string}.pdf"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()


# %% rqa measures vs. catchment size

fig, ax = plt.subplots(1, 1, figsize=(6, 3))

for i in range(len(images)):

    cm_nr = name_list[i].split("_")[0]
    datatype = path.split("/")[-1].split("_")[0]
    filepath_slp = f"data/{datatype}"

    dataset_slp = rio.open(os.path.join(filepath_slp, f"{cm_nr}_lin_slp.tif"))
    slp = dataset_slp.read().squeeze()

    val = slp.size

    colname = [col for col in df.columns if col.startswith("intercept")][0]

    if name_list[i] in list(df.name):
        ax.scatter(
            val, df[df.name == name_list[i]].det_line_rot, c="royalblue", s=70, alpha=0.7
        )
        ax.scatter(
            val, df[df.name == name_list[i]].lam_line_rot, c="crimson", s=70, alpha=0.7
        )
        ax.scatter(val, df[df.name == name_list[i]][colname], c="darkorange", s=70, alpha=0.7)

ax.set_ylim(-0.1, 1.5)
ax.set_xlabel("catchment size [px]", fontsize=10, labelpad=10)
ax.set_ylabel(f"RQA measure value", fontsize=10, labelpad=10)
ax.tick_params(axis="both", labelsize=8)
ax.grid()
ax.set_axisbelow(True)
ax.set_title(
    f"Lacunarity - Catchment size and RQA Measures",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

handles, labels = plt.gca().get_legend_handles_labels()

point1 = Line2D(
    [0],
    [0],
    label="Determinism",
    marker="o",
    markersize=8,
    c="royalblue",
    linestyle="",
    alpha=0.7,
)

point2 = Line2D(
    [0],
    [0],
    label="Laminarity",
    marker="o",
    markersize=8,
    c="crimson",
    linestyle="",
    alpha=0.7,
)

point3 = Line2D(
    [0],
    [0],
    label="Lacunarity at w = 10",
    marker="o",
    markersize=8,
    c="darkorange",
    linestyle="",
    alpha=0.7,
)

handles.extend([point1, point2, point3])

ax.legend(handles=handles, fontsize=10, framealpha=1, edgecolor="k", borderpad=0.6)
#plt.show()
plt.savefig(os.path.join(path, f"grqa_size.png"), bbox_inches="tight", dpi=300)
plt.savefig(os.path.join(path, f"grqa_size.pdf"), bbox_inches="tight", dpi=300)
plt.close()


#%% white noise variability

fig, ax_grid = plt.subplots(
        2, 3, figsize=(10, 5), gridspec_kw={"height_ratios": [2, 1]}
    )

for j, size in enumerate(["500", "750", "1000"]):

    ax = ax_grid[0, j]
    axh = ax_grid[1, j]
    #axbar = ax_grid[2, j]

    dirlist = [
            d
            for d in os.listdir(path)
            if ((os.path.isdir(os.path.join(path, d)) and d.startswith(size)))
        ]

    filelist = [os.path.join(path, d, f"{d}_mass_distribution.csv") for d in dirlist]

    arr_lac = np.ones((301, 10)) * np.nan

    for i, file in enumerate(filelist):
        try:
            df_lac = pd.read_csv(file)
            arr_lac[:, i] = df_lac.lacunarity
        except:
            continue

    # plot all lacunarities
    for i in range(arr_lac.shape[1]):
        x = np.where(~np.isnan(arr_lac[:, 0]))[0]
        if i == 0:
            ax.loglog(x, arr_lac[x, i], c="silver",
                  label=f"all reali-\nzations")
        else:
            ax.loglog(x, arr_lac[x, i], c="silver")

    # calculate & plot lac mean
    arr_mean = np.nanmean(arr_lac, axis=1)
    x = np.where(~np.isnan(arr_mean))[0]
    ax.loglog(x, arr_mean[x], c="k", lw=2, linestyle="--", label="mean")

    # calculate & plot residuals
    arr_resid = np.abs(np.log10(arr_lac) - np.log10(arr_mean[:,None]))
    axh.plot(x, np.nanmean(arr_resid, axis=1)[x], c="navy", label="mean")
    print("###", size, "###")
    print(x[-12:])
    print(arr_mean[x][-12:])
    print(np.nanmean(arr_resid, axis=1)[x][-12:])
    #axh.plot(x, np.nanpercentile(arr_resid, 10, axis=1)[x], linestyle="--", c="b",
    #         label="10th percentile")
    #axh.plot(x, np.nanpercentile(arr_resid, 90, axis=1)[x], linestyle="-.", c="b",
    #         label="90th percentile")
    axh.fill_between(x,
                     np.nanpercentile(arr_resid, 10, axis=1)[x],
                     np.nanpercentile(arr_resid, 90, axis=1)[x],
                     color="navy", alpha=0.5, lw=0,
                     label=f"10th to 90th\npercentile")

    # calculate & plot residual distribution
    #be = np.arange(0, 1.5, 0.05)
    #counts, bins = np.histogram(arr_resid.flatten(), bins=be)
    #rel_freq = counts / counts.sum()

    #percentile_90 = 0.07 #np.nanpercentile(arr_resid.flatten(), 85)
    #axbar.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge', edgecolor='black',
    #       facecolor="lightskyblue")
    #axbar.axvline(percentile_90, lw=2, c="k", linestyle="-.", label=f"90th perc.:\n{percentile_90:.1f}")

    #axh.axhline(percentile_90)
    #idx_90 = np.where(np.nanmean(arr_resid, axis=1) > percentile_90)[0][0]
    #axh.axvline(idx_90)
    #ax.axvline(idx_90)

    ax.set_xlim(0.7, 350)
    ax.set_ylim(0.5e-8, 1e1)
    ax.set_xticklabels("")

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_title(f"{size} x {size} px", fontweight="bold", fontsize=14)

    axh.set_xlim(0.7, 350)
    axh.set_ylim(0, 1.2)
    axh.set_xscale('log')

    axh.grid()
    axh.set_axisbelow(True)

    #axbar.set_ylim(0, 1)

    if j > 0:
        ax.set_yticklabels("")
        axh.set_yticklabels("")
    else:
        ax.set_ylabel("lacunarity", fontsize=12)
        ax.legend(loc="lower left", fontsize=10, framealpha=1,
                  edgecolor="k",
                  borderpad=0.6)
        axh.set_ylabel("abs. log. residuals", fontsize=12)
        axh.legend(loc="upper left", fontsize=10, framealpha=1,
                   edgecolor="k",
                   borderpad=0.6)

fig.suptitle(f"White Noise - Lacunarity", fontsize=18, fontweight="bold")

fig.subplots_adjust(top=0.85, hspace=0.1, wspace=0.05)

#plt.show()
plt.savefig(
    os.path.join(path, f"lacunarity_comparison_residuals.png"), bbox_inches="tight", dpi=300
)
plt.savefig(
    os.path.join(path, f"lacunarity_comparison_residuals.pdf"), bbox_inches="tight", dpi=300
)
plt.close()


# %% create latex display for all images in the directory

imglist = []

orig_stdout = sys.stdout
tex = open(os.path.join(path, "plot_overview.tex"), "w")
sys.stdout = tex

print(
    """
\\documentclass[11pt,twoside,ngerman, BCOR = 1cm, DIV = 10]{scrreprt}

\\usepackage[a4paper, left=2cm, right=2cm, top=2cm, bottom=1cm]{geometry}
\\usepackage{pdflscape}

% Alles zu Sprache
\\usepackage[utf8]{inputenc}
\\usepackage{babel}
\\usepackage[T1]{fontenc}
\\usepackage{hyphenat} 
\\hyphenation{Mathe-matik wieder-gewinnen}

% Alles zu Mathe
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{makeidx}

% Alles zu Graphiken
\\usepackage{caption}
\\usepackage{subcaption}
\\captionsetup{format=plain} 
\\usepackage{graphicx}
\\usepackage{xcolor}
\n
"""
)

print("\\begin{document}")
print("\n")
print("\\tableofcontents")
print("\\newpage")

width = 0.5

for n in name_list:

    if not os.path.isdir(join(path, n)):
        continue

    print("\\subsection{" + n.replace("_", " ") + "}")
    print("\n")
    filelist = [
        f
        for f in os.listdir(join(path, n))
        if (os.path.isfile(join(path, n, f)) and f.endswith(".png"))
    ]
    filelist = sorted(filelist)
    for f in filelist:
        print(
            "\\includegraphics[width=" + str(width) + "\\textwidth]{" + join(n, f) + "}"
        )
    print("\\newpage")
    print("\n")

print("\\begin{landscape}")
print("\\subsection{GRQA measures}")
print("\n")
filelist = [
    f
    for f in os.listdir(path)
    if (
        os.path.isfile(join(path, f))
        and f.endswith(".png")
        and (f.startswith("grqa") or f.startswith("lac"))
    )
]
for f in filelist:
    print("\\includegraphics[width=1.6\\textwidth]{" + f + "}")
    print("\\newpage")
print("\\end{landscape}")

print("\\end{document}")

sys.stdout = orig_stdout
tex.close()


# %% plot lacunarity of 1m and 2m together

path1 = "img/slp_1m"
path2 = "img/slp_2m"
minlen = 2
peak_prominence = 0.05

images, _, name_list = utils.collect_slp()

for i in trange(len(images)):
    try:
        mass_dist_df1 = pd.read_csv(
            os.path.join(path1, name_list[i], f"{name_list[i]}_mass_distribution.csv")
        )
        mass_dist_df2 = pd.read_csv(
            os.path.join(path2, name_list[i], f"{name_list[i]}_mass_distribution.csv")
        )
    except FileNotFoundError:
        continue

    lac1 = mass_dist_df1.lacunarity
    lac1[lac1 == 0] = np.nan
    bs1 = mass_dist_df1.box_size

    lac2 = mass_dist_df2.lacunarity
    lac2[lac2 == 0] = np.nan
    bs2 = mass_dist_df2.box_size

    ### lacunarity

    bs1 = bs1[(~np.isnan(lac1)) & (lac1 > 0)].to_numpy()
    lac1 = lac1[(~np.isnan(lac1)) & (lac1 > 0)].to_numpy()

    bs2 = bs2[(~np.isnan(lac2)) & (lac2 > 0)].to_numpy()
    lac2 = lac2[(~np.isnan(lac2)) & (lac2 > 0)].to_numpy()

    idx_max1, _ = find_peaks(np.log10(lac1), prominence=peak_prominence)
    idx_min1, _ = find_peaks(-np.log10(lac1), prominence=peak_prominence)

    idx_max2, _ = find_peaks(np.log10(lac2), prominence=peak_prominence)
    idx_min2, _ = find_peaks(-np.log10(lac2), prominence=peak_prominence)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.loglog(bs1, lac1, c="darkmagenta", lw=2, label="LAC (1m)")
    ax.loglog(bs2, lac2, c="forestgreen", lw=2, label="LAC (2m)")

    if idx_min1.size != 0:
        ax.scatter(
            bs1[idx_min1],
            lac1[idx_min1],
            label=f"Minima at: {bs1[idx_min1]}",
            c="darkmagenta",
            alpha=0.5,
        )

    if idx_max1.size != 0:
        ax.scatter(
            bs1[idx_max1],
            lac1[idx_max1],
            c="navy",
            alpha=0.5,
            label=f"Maxima at: {bs1[idx_max1]}",
        )

    if idx_min2.size != 0:
        ax.scatter(
            bs2[idx_min2],
            lac2[idx_min2],
            label=f"Minima at: {bs2[idx_min2]}",
            c="forestgreen",
            alpha=0.5,
        )

    if idx_max2.size != 0:
        ax.scatter(
            bs2[idx_max2],
            lac2[idx_max2],
            c="lime",
            alpha=0.5,
            label=f"Maxima at: {bs2[idx_max2]}",
        )

    ax.set_ylabel("Lacunarity")
    ax.set_xlabel("Box size (w)")
    ax.grid()
    ax.set_axisbelow(True)
    ax.set_title(f"{name_list[i]}", fontsize=14, fontweight="bold", pad=10)

    ax.legend()

    plt.savefig(
        os.path.join(path1, name_list[i], "lacunarity_compare.png"),
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()
    plt.close()


# %% all lacunarity curves

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for i in range(len(images)):
    try:
        mass_dist_df1 = pd.read_csv(
            os.path.join(path1, name_list[i], f"{name_list[i]}_mass_distribution.csv")
        )
        mass_dist_df2 = pd.read_csv(
            os.path.join(path2, name_list[i], f"{name_list[i]}_mass_distribution.csv")
        )
    except FileNotFoundError:
        continue

    lac1 = mass_dist_df1.lacunarity
    lac1[lac1 == 0] = np.nan
    bs1 = mass_dist_df1.box_size

    lac2 = mass_dist_df2.lacunarity
    lac2[lac2 == 0] = np.nan
    bs2 = mass_dist_df2.box_size

    bs1 = bs1[(~np.isnan(lac1)) & (lac1 > 0)].to_numpy()
    lac1 = lac1[(~np.isnan(lac1)) & (lac1 > 0)].to_numpy()

    bs2 = bs2[(~np.isnan(lac2)) & (lac2 > 0)].to_numpy()
    lac2 = lac2[(~np.isnan(lac2)) & (lac2 > 0)].to_numpy()

    ax.loglog(bs1, lac1, c="darkmagenta", lw=2, alpha=0.7)
    ax.loglog(bs2, lac2, c="forestgreen", lw=2, alpha=0.7)


ax.set_ylabel("Lacunarity")
ax.set_xlabel("Box size (w)")
ax.grid()
ax.set_axisbelow(True)
ax.set_title(f"Lacunarity Curves", fontsize=14, fontweight="bold", pad=10)

point1 = Line2D([0], [0], label="LAC (1m)", alpha=0.7, lw=2, c="darkmagenta")

point2 = Line2D([0], [0], label="LAC (2m)", alpha=0.7, lw=2, c="forestgreen")


handles, labels = plt.gca().get_legend_handles_labels()

handles.extend([point1, point2])

ax.legend(handles=handles)
# plt.show()
plt.savefig(
    os.path.join(path1, f"lacunarity_all_resolution.png"), bbox_inches="tight", dpi=300
)
plt.close()


#%% manual bacteria image measurements

for name in name_list:
    im = plt.imread(os.path.join(path, name, f"{name}_original.png"))
    lines = pd.read_csv(os.path.join(path, name, f"Results_new.csv"))
    lines["Length"] = lines.Length.astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(5, 6))

    dpi = 100
    height, width = img.shape[:2]

    plt.close("all")
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    plt.imshow(im, cmap=cmap, vmin=-0.5, vmax=3.5, rasterized=True)
    plt.axis("off")
    plt.tight_layout(pad=0)

    for idx in lines.index:
        l = lines.loc[idx]
        if (name == "8") and (idx == 9):
            plt.text(l.BX, l.BY, f"{int(l.Length)}", fontweight="bold", fontsize=16,
                     verticalalignment="bottom",
                     horizontalalignment="left")
        else:
            plt.text(l.BX + l.Width + 2, l.BY + l.Height / 2, f"{int(l.Length)}", fontweight="bold", fontsize=16,
                     verticalalignment="bottom",
                     horizontalalignment="left")

    #plt.show()
    plt.savefig(os.path.join(path, name, name + "_original_with_lengths.png"), dpi=100, bbox_inches="tight", pad_inches=0)
    plt.savefig(os.path.join(path, name, name + "_original_with_lengths.pdf"), dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close("all")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 1))

    lines_df = pd.DataFrame({"mean": f"{int(np.round(lines.Length.mean(), 0))} px",
                             "std": f"{int(np.round(lines.Length.std(), 0))} px"}, index=[0])
    lines_df2 = pd.DataFrame({"mean / 3": f"{int(np.round(lines.Length.mean() / 3, 0))} px",
                             "std / 3": f"{int(np.round(lines.Length.std() / 3, 0))} px"}, index=[0])

    t1 = ax1.table(lines_df.values, colLabels=lines_df.columns, loc="center", fontsize=16)
    ax1.set_axis_off()
    t1.scale(1, 1.5)
    t1.auto_set_font_size(False)
    for (i, j), cell in t1.get_celld().items():
        if i == 0:
            # Make column labels (header row) bold
            cell.set_text_props(weight='bold')

    t2 = ax2.table(lines_df2.values, colLabels=lines_df2.columns, loc="center", fontsize=16)
    ax2.set_axis_off()
    t2.scale(1, 1.5)
    t2.auto_set_font_size(False)
    for (i, j), cell in t2.get_celld().items():
        if i == 0:
            # Make column labels (header row) bold
            cell.set_text_props(weight='bold')

    #plt.show()
    plt.savefig(os.path.join(path, name, name + "table.png"), dpi=100, bbox_inches="tight", pad_inches=0)
    plt.savefig(os.path.join(path, name, name + "table.pdf"), dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close()