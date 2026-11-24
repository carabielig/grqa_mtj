import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import rasterio as rio
from matplotlib import cm
from scipy.spatial.distance import pdist, squareform
from skimage.transform import downscale_local_mean

import kaleido

# TODO: save from fixed camera position
def plot_rp_3d_from_rp(rp, plot_path, name="rp", cmap="Greys_r", origin="lower"):
    """
    Plot 3d subset of a recurrence plot (or other 4d array, e.g. distance matrix) from the whole recurrence
    plot (does not work if the recurrence plot does not fit into memory).
    """

    # thresholds for the coloring?
    cmin = rp.min()
    cmax = rp.max()

    # prepare coordinate tuples for both surfaces
    xx_hor, yy_hor = np.meshgrid(list(range(rp.shape[0])), list(range(rp.shape[0])))
    xx_vert, zz_vert = np.meshgrid(list(range(rp.shape[0])), list(range(rp.shape[1])))

    x1 = int(rp.shape[1] / 2)
    # y2 = int(rp.shape[0] - 10)
    y2 = int(rp.shape[0] - 1)

    rp2d = np.zeros((rp.shape[1], rp.shape[0]))

    for x in range(rp.shape[1]):
        for y in range(rp.shape[0]):
            row = rp[:, x, y2, x]
            rp2d[x, :] = row
    rp2d = rp2d.astype(np.int32)

    fig = go.Figure(
        data=[
            go.Surface(
                x=xx_hor,
                y=yy_hor,
                z=np.ones(yy_hor.shape) * x1,
                surfacecolor=rp[:, x1, :, x1],
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax,
            ),
            go.Surface(
                x=xx_vert,
                y=np.ones(zz_vert.shape) * y2,
                z=zz_vert,
                surfacecolor=rp2d[:, :],
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax,
            ),
        ]
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        scene=dict(xaxis_title="y1", yaxis_title="y2", zaxis_title="x1 = x2")
    )

    fig.write_html(os.path.join(plot_path, name + ".html"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    cmap = cm.get_cmap(cmap, 2)

    im1 = ax1.imshow(
        rp[:, x1, :, x1],
        origin=origin,
        interpolation="None",
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        rasterized=True,
    )
    ax1.set_xlabel("y1", labelpad=10, fontsize=14)
    ax1.set_ylabel("y2", labelpad=10, fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_title(f"x1 = x2 = {x1}", fontweight="bold", fontsize=16, pad=10)

    im2 = ax2.imshow(
        np.swapaxes(rp2d, 0, 1),
        origin=origin,
        interpolation="None",
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        rasterized=True,
    )
    ax2.set_xlabel("x1 = x2", labelpad=10, fontsize=14)
    ax2.set_ylabel("y1", labelpad=10, fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.set_title(f"y2 = {y2}", fontweight="bold", fontsize=16, pad=10)

    # ax1.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])
    # ax2.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])

    cbar1 = plt.colorbar(im1, fraction=0.046, ticks=[0.25, 0.75])
    cbar1.set_label("recurrence", size=14)
    cbar1.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})
    cbar2 = plt.colorbar(im2, fraction=0.046, ticks=[0.25, 0.75])
    cbar2.set_label("recurrence", size=14)
    cbar2.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})

    plt.subplots_adjust(wspace=0.6)

    plt.savefig(os.path.join(plot_path, name + "_3d.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(plot_path, name + "_3d.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_rp_3d(img, epsilon, plot_path, name="rp", cmap="Greys_r", col=None, origin="lower"):
    """
    Plot 3d subset of a recurrence plot (or other 4d array, e.g. distance matrix). Only computes the horizontal
    and vertical plane for the visualization (also works if rp does not fit into memory).
    """

    ys, xs = img.shape

    horizontal_plane = np.zeros((ys, ys))
    vertical_plane = np.zeros((xs, ys))

    if not col:
        x_hor = int(xs / 2)
    else:
        x_hor = col
    col = img[:, x_hor]
    dm = squareform(pdist(np.c_[col, col]))
    horizontal_plane[dm <= epsilon] = 1

    y_vert = ys - 1
    for x in range(xs):
        col = img[:, x]
        px = img[y_vert, x]
        vertical_plane[x, :] = (np.abs(col - px) <= epsilon).astype(int)

    # thresholds for the coloring?
    cmin = 0  # rp.min()
    cmax = 1  # rp.max()

    # prepare coordinate tuples for both surfaces
    xx_hor, yy_hor = np.meshgrid(list(range(ys)), list(range(ys)))
    xx_vert, zz_vert = np.meshgrid(list(range(ys)), list(range(xs)))

    fig = go.Figure(
        data=[
            go.Surface(
                x=xx_hor,
                y=yy_hor,
                z=np.ones(yy_hor.shape) * x_hor,
                surfacecolor=horizontal_plane,
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax,
            ),
            go.Surface(
                x=xx_vert,
                y=np.ones(zz_vert.shape) * y_vert,
                z=zz_vert,
                surfacecolor=vertical_plane,
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax,
            ),
        ]
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        scene=dict(xaxis_title="y1", yaxis_title="y2", zaxis_title="x1 = x2")
    )

    fig.write_html(os.path.join(plot_path, name + ".html"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    cmap = cm.get_cmap(cmap, 2)

    im1 = ax1.imshow(
        horizontal_plane,
        origin=origin,
        interpolation="None",
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        rasterized=True,
    )
    ax1.set_xlabel("y1", labelpad=10, fontsize=14)
    ax1.set_ylabel("y2", labelpad=10, fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_title(f"x1 = x2 = {x_hor}", fontweight="bold", fontsize=16, pad=10)

    im2 = ax2.imshow(
        np.swapaxes(vertical_plane, 0, 1),
        origin=origin,
        interpolation="None",
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        rasterized=True,
    )
    ax2.set_xlabel("x1 = x2", labelpad=10, fontsize=14)
    ax2.set_ylabel("y1", labelpad=10, fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.set_title(f"y2 = {y_vert}", fontweight="bold", fontsize=16, pad=10)

    # ax1.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])
    # ax2.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])

    cbar1 = plt.colorbar(im1, fraction=0.046, ticks=[0.25, 0.75])
    cbar1.set_label("recurrence", size=14)
    cbar1.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})
    cbar2 = plt.colorbar(im2, fraction=0.046, ticks=[0.25, 0.75])
    cbar2.set_label("recurrence", size=14)
    cbar2.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})

    plt.subplots_adjust(wspace=0.6)

    plt.savefig(os.path.join(plot_path, name + "_3d.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(plot_path, name + "_3d.pdf"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_rp_3d_new(
    img,
    epsilon,
    plot_path,
    name="rp",
    cmap="Greys_r",
    col=None,
    row=None,
    origin="lower",
    only_2d=False,
):
    """
    Plot 3d subset of a recurrence plot (or other 4d array, e.g. distance matrix). Only computes the horizontal
    and vertical plane for the visualization (also works if rp does not fit into memory).
    """

    ys, xs = img.shape

    horizontal_plane = np.zeros((ys, ys))
    vertical_plane = np.zeros((xs, ys))

    if not col:
        x_hor = int(xs / 2)
    else:
        x_hor = col

    col = img[:, x_hor]
    dm = squareform(pdist(np.c_[col, col]))
    horizontal_plane[dm <= epsilon] = 1

    if not row:
        y_vert = ys - 1
    else:
        y_vert = row

    for x in range(xs):
        col = img[:, x]
        px = img[y_vert, x]
        vertical_plane[x, :] = (np.abs(col - px) <= epsilon).astype(int)

    # thresholds for the coloring?
    cmin = 0  # rp.min()
    cmax = 1  # rp.max()

    # prepare coordinate tuples for both surfaces
    xx_hor, yy_hor = np.meshgrid(list(range(ys)), list(range(ys)))
    xx_vert, zz_vert = np.meshgrid(list(range(ys)), list(range(xs)))

    if cmap == "Greys_r":
        binary_colorscale = [
            [0.0, "rgb(0,0,0)"],
            [0.5, "rgb(0,0,0)"],
            [0.5, "rgb(255,255,255)"],
            [1.0, "rgb(255,255,255)"],
        ]
    else:
        binary_colorscale = [
            [0.0, "rgb(255,255,255)"],
            [0.5, "rgb(255,255,255)"],
            [0.5, "rgb(0,0,0)"],
            [1.0, "rgb(0,0,0)"],
        ]

    if only_2d == False:

        fig = go.Figure(
            data=[
                go.Surface(
                    x=xx_hor,
                    y=yy_hor,
                    z=np.ones(yy_hor.shape) * x_hor,
                    surfacecolor=horizontal_plane,
                    colorscale=binary_colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(
                        x=1,
                        title=dict(
                            text="recurrence", font=dict(size=24, color="rgb(0, 0, 0)")
                        ),
                        tickvals=[0.25, 0.75],
                        ticktext=["0", "1"],
                        tickfont=dict(size=20, color="rgb(0, 0, 0)"),
                        len=0.3,
                        xanchor="left",
                        yanchor="middle",
                        outlinecolor="black",
                        outlinewidth=2,
                    ),
                ),
                go.Surface(
                    x=xx_vert,
                    y=np.ones(zz_vert.shape) * y_vert,
                    z=zz_vert,
                    surfacecolor=vertical_plane,
                    colorscale=binary_colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    showscale=False,
                ),
            ]
        )

        fig.update_yaxes(autorange="reversed")

        camera_eye = dict(x=-1.7, y=-1.7, z=1)

        fig.update_layout(
            scene=dict(
                xaxis_title=dict(text="y1", font=dict(size=22, color="rgb(0, 0, 0)")),
                yaxis_title=dict(text="y2", font=dict(size=22, color="rgb(0, 0, 0)")),
                zaxis_title=dict(
                    text="x1 = x2", font=dict(size=22, color="rgb(0, 0, 0)")
                ),
                camera=dict(
                    up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=camera_eye
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            font=dict(size=14, color="rgb(0, 0, 0)"),
        )

        fig.write_html(os.path.join(plot_path, name + "_3d.html"))
        fig.write_image(
            os.path.join(plot_path, name + "_rp3d.png"), width=850, height=800
        )
        fig.write_image(
            os.path.join(plot_path, name + "_rp3d.pdf"), width=850, height=800
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        cmap = cm.get_cmap(cmap, 2)

        im1 = ax1.imshow(
            horizontal_plane,
            origin=origin,
            interpolation="None",
            cmap=cmap,
            vmin=cmin,
            vmax=cmax,
            rasterized=True,
        )
        ax1.set_xlabel("y1", labelpad=10, fontsize=14)
        ax1.set_ylabel("y2", labelpad=10, fontsize=14)
        ax1.tick_params(axis="both", labelsize=12)
        ax1.set_title(f"x1 = x2 = {x_hor}", fontweight="bold", fontsize=16, pad=10)

        im2 = ax2.imshow(
            np.swapaxes(vertical_plane, 0, 1),
            origin=origin,
            interpolation="None",
            cmap=cmap,
            vmin=cmin,
            vmax=cmax,
            rasterized=True,
        )
        ax2.set_xlabel("x1 = x2", labelpad=10, fontsize=14)
        ax2.set_ylabel("y1", labelpad=10, fontsize=14)
        ax2.tick_params(axis="both", labelsize=12)
        ax2.set_title(f"y2 = {y_vert}", fontweight="bold", fontsize=16, pad=10)

        # ax1.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])
        # ax2.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])

        cbar1 = plt.colorbar(im1, fraction=0.046, ticks=[0.25, 0.75])
        cbar1.set_label("recurrence", size=14)
        cbar1.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})
        cbar2 = plt.colorbar(im2, fraction=0.046, ticks=[0.25, 0.75])
        cbar2.set_label("recurrence", size=14)
        cbar2.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})

        plt.subplots_adjust(wspace=0.6)

        plt.savefig(
            os.path.join(plot_path, name + "_rp.png"), dpi=300, bbox_inches="tight"
        )
        plt.savefig(
            os.path.join(plot_path, name + "_rp.pdf"), dpi=300, bbox_inches="tight"
        )
        plt.close()



def plot_rp_3d_rev(img, epsilon, plot_path, name="rp", cmap="Greys_r", row=None, origin="lower"):
    """
    Switch the horizontal (x1 = x2 --> y1 = y2) and vertical plane (y2 = a --> x2 = a).
    """

    img = np.rot90(img)

    ys, xs = img.shape

    horizontal_plane = np.zeros((ys, ys))
    vertical_plane = np.zeros((xs, ys))

    if not row:
        x_hor = int(xs / 2)
    else:
        x_hor = row
    col = img[:, x_hor]
    dm = squareform(pdist(np.c_[col, col]))
    horizontal_plane[dm <= epsilon] = 1

    y_vert = ys - 1
    for x in range(xs):
        col = img[:, x]
        px = img[y_vert, x]
        vertical_plane[x, :] = (np.abs(col - px) <= epsilon).astype(int)

    # thresholds for the coloring?
    cmin = 0  # rp.min()
    cmax = 1  # rp.max()

    # prepare coordinate tuples for both surfaces
    xx_hor, yy_hor = np.meshgrid(list(range(ys)), list(range(ys)))
    xx_vert, zz_vert = np.meshgrid(list(range(ys)), list(range(xs)))

    fig = go.Figure(
        data=[
            go.Surface(
                x=xx_hor,
                y=yy_hor,
                z=np.ones(yy_hor.shape) * x_hor,
                surfacecolor=horizontal_plane,
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax,
            ),
            go.Surface(
                x=xx_vert,
                y=np.ones(zz_vert.shape) * y_vert,
                z=zz_vert,
                surfacecolor=vertical_plane,
                colorscale=cmap,
                cmin=cmin,
                cmax=cmax,
            ),
        ]
    )

    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="y1 = y2")
    )

    fig.write_html(os.path.join(plot_path, name + "_rev.html"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    cmap = cm.get_cmap(cmap, 2)

    im1 = ax1.imshow(
        horizontal_plane,
        origin=origin,
        interpolation="None",
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        rasterized=True,
    )
    ax1.set_xlabel("x1", labelpad=10, fontsize=14)
    ax1.set_ylabel("x2", labelpad=10, fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_title(f"y1 = y2 = {x_hor}", fontweight="bold", fontsize=16, pad=10)

    im2 = ax2.imshow(
        vertical_plane,
        origin=origin,
        interpolation="None",
        cmap=cmap,
        vmin=cmin,
        vmax=cmax,
        rasterized=True,
    )
    ax2.set_xlabel("x1", labelpad=10, fontsize=14)
    ax2.set_ylabel("y1 = y2", labelpad=10, fontsize=14)
    ax2.tick_params(axis="both", labelsize=12)
    ax2.set_title(f"x2 = {y_vert}", fontweight="bold", fontsize=16, pad=10)

    # ax1.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])
    # ax2.set_yticks([0, 5, 10, 15], [0, 5, 10, 15])

    cbar1 = plt.colorbar(im1, fraction=0.046, ticks=[0.25, 0.75])
    cbar1.set_label("recurrence", size=14)
    cbar1.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})
    cbar2 = plt.colorbar(im2, fraction=0.046, ticks=[0.25, 0.75])
    cbar2.set_label("recurrence", size=14)
    cbar2.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 12})

    plt.subplots_adjust(wspace=0.6)

    plt.savefig(
        os.path.join(plot_path, name + "_3d_y.png"), dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(plot_path, name + "_3d_y.pdf"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def load_and_plot_landsat_rgb(
    axis, prefix="LC08_L2SP_046032_20250805_20250815_02_T1_SR", path="data/mtj_landsat"
):

    def normalize(b):
        b = downscale_local_mean(b, 20)
        b = b / 15000
        # b = b / (np.mean(b) + 2 * np.std(b))
        b[b > 1] = 1
        return b

    b2_dataset = rio.open(os.path.join(path, prefix + "_B2.TIF"))
    b2 = b2_dataset.read().squeeze()
    b2 = normalize(b2)
    b3 = rio.open(os.path.join(path, prefix + "_B3.TIF"))
    b3 = b3.read().squeeze()
    b3 = normalize(b3)
    b4 = rio.open(os.path.join(path, prefix + "_B4.TIF"))
    b4 = b4.read().squeeze()
    b4 = normalize(b4)

    landsat_rgb = np.stack([b4, b3, b2], axis=2)
    landsat_rgb[
        (landsat_rgb[:, :, 0] == 0)
        & (landsat_rgb[:, :, 1] == 0)
        & (landsat_rgb[:, :, 2] == 0)
    ] = np.nan
    landsat_rgb_extent = np.array(b2_dataset.bounds)[[0, 2, 1, 3]]

    axis.imshow(landsat_rgb, extent=landsat_rgb_extent, rasterized=True)


def plot_hist(
    diag_frequ,
    diag_len,
    diag_dir_fac,
    vert_frequ,
    vert_len,
    vert_dir_fac,
    suptitle,
    filepath,
):

    edgecolor = "teal"
    facecolor = "teal"

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].bar(
        diag_len,
        diag_frequ / diag_dir_fac,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=0,
    )
    ylim = np.max(np.c_[diag_frequ / diag_dir_fac, vert_frequ / vert_dir_fac])
    ax[0].set_ylim((0, ylim + ylim * 0.05))
    ax[0].set_ylabel("absolute frequency", fontsize=12)
    ax[0].set_xlabel("length", fontsize=12)
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[0].set_title("Diagonal", fontsize=14, fontweight="bold", pad=10)

    ax[1].bar(
        vert_len,
        vert_frequ / vert_dir_fac,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=0,
    )
    ax[1].set_ylim((0, ylim + ylim * 0.05))
    ax[1].set_ylabel("absolute frequency", fontsize=12)
    ax[1].set_xlabel("length", fontsize=12)
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[1].set_title("Vertical", fontsize=14, fontweight="bold", pad=10)

    fig.suptitle(suptitle, fontweight="bold", fontsize=18)
    fig.subplots_adjust(top=0.8, wspace=0.3)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.savefig(filepath.split(".")[0] + ".pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_hist_mass(
    bs,
    mass_distribution,
    mass_distribution_z2,
    suptitle,
    filepath,
    add_longitudinal=False,
    data_label="",
):

    edgecolor = "darkmagenta"
    facecolor = "darkmagenta"

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].bar(
        bs,
        mass_distribution,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=0,
    )
    ylim = np.max(mass_distribution)
    ax[0].set_ylim((0, ylim + ylim * 0.05))
    ax[0].set_ylabel("sum of masses", fontsize=12)
    ax[0].set_xlabel("box size (w)", fontsize=12)
    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[0].set_title("Sum of Masses", fontsize=14, fontweight="bold", pad=10)

    ax[1].bar(
        bs,
        mass_distribution_z2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=0,
    )
    ylim = np.max(mass_distribution_z2)
    ax[1].set_ylim((0, ylim + ylim * 0.05))
    ax[1].set_ylabel("sum of squared masses", fontsize=12)
    ax[1].set_xlabel("box size (w)", fontsize=12)
    ax[1].grid()
    ax[1].set_axisbelow(True)
    ax[1].set_title("Sum of Squared Masses", fontsize=14, fontweight="bold", pad=10)

    if add_longitudinal:
        mass_dist_df_per = pd.read_csv(
            os.path.join(
                "img/test_images", "longitudinal", f"longitudinal_mass_distribution.csv"
            )
        )

        mass_dist_df_per.loc[
            100,
            [
                "mass_distribution",
                "mass_distribution_reverse",
                "mass_distribution_z2",
                "mass_distribution_reverse_z2",
            ],
        ] = (
            mass_dist_df_per.loc[
                100,
                [
                    "mass_distribution",
                    "mass_distribution_reverse",
                    "mass_distribution_z2",
                    "mass_distribution_reverse_z2",
                ],
            ]
            / 2
        )

        ax[0].bar(
            bs,
            mass_distribution,
            edgecolor=edgecolor,
            facecolor="thistle",
            linewidth=0,
            label=data_label,
        )
        ax[0].plot(
            bs, mass_distribution, c=facecolor, linewidth=3, alpha=0.7, label=data_label
        )
        ax[0].plot(
            bs,
            mass_dist_df_per.mass_distribution,
            c="deepskyblue",
            linewidth=3,
            alpha=0.5,
            label="Logitudinal Wave",
        )
        ax[1].bar(
            bs,
            mass_distribution_z2,
            edgecolor=edgecolor,
            facecolor="thistle",
            linewidth=0,
            label=data_label,
        )
        ax[1].plot(
            bs,
            mass_distribution_z2,
            c=facecolor,
            linewidth=3,
            alpha=0.7,
            label=data_label,
        )
        ax[1].plot(
            bs,
            mass_dist_df_per.mass_distribution_z2,
            c="deepskyblue",
            linewidth=3,
            alpha=0.7,
            label="Logitudinal Wave",
        )

        plt.legend(fontsize=10, framealpha=1, edgecolor="k", borderpad=0.6)

    fig.suptitle(suptitle, fontweight="bold", fontsize=18)
    fig.subplots_adjust(top=0.8, wspace=0.2)

    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.savefig(filepath.split(".")[0] + ".pdf", dpi=300, bbox_inches="tight")
    plt.close()
