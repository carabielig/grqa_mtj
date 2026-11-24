import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.spatial.distance import pdist, squareform

from scripts import grqa, utils


def create_flower_image():
    flower = (np.arange(100, 10100, 100)).reshape((10, 10))

    # stem
    flower[0:5, 4] = 0

    # left leaf
    flower[0:2, 3] = 0
    flower[1, 2] = 0
    flower[2:4, 1:3] = 0

    # right leaf
    flower[1:3, 5] = 0
    flower[2:3, 6] = 0
    flower[3:5, 7:9] = 0

    # blossom
    flower[5:10, 3:6] = 0
    flower[6:9, 2:7] = 0

    return flower


# %% load example image

### flower image example
# im = create_flower_image()

### turing pattern example
# im = utils.get_turing_test_image(20)
# np.save("img/test_patterns/turing_example.npy", Z_turing)
# load turing pattern example
im = np.load("img/latex/turing_example.npy")


### logistic map pattern example
# im = utils.get_logistic_map_test_image(20)


### random noise test pattern
# size = 20
# im = np.random.uniform(0, 1, (size, size))


# %% plot test image

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

cb = ax.imshow(im, origin="lower", cmap="Blues_r", rasterized=True)
ax.set_xlabel("x", fontsize=18, labelpad=10)
ax.set_ylabel("y", fontsize=18, labelpad=10)
ax.tick_params(axis="both", labelsize=16)
ax.set_xticks(np.arange(0, 20, 2))
ax.set_yticks(np.arange(0, 20, 2))
cbar = plt.colorbar(cb, ax=ax, fraction=0.046, label="pixel value")
cbar.set_label("pixel value", fontsize=18, labelpad=10)
cbar.ax.tick_params(labelsize=16)

plt.show()
#plt.savefig("img/latex/turing_example_blue.png", dpi=300, bbox_inches="tight")
#plt.savefig("img/latex/turing_example_blue.pdf", dpi=300, bbox_inches="tight")
plt.close()


# %% generate rp for test image

dm = grqa.distance_matrix(im)
epsilon = grqa.recurrence_threshold(dm, 0.3)
rp = grqa.recurrence_plot(dm, epsilon)

#utils.plot_rp_3d_from_rp(rp, plot_path="img/latex/", name=f"turing_example_3drp")
utils.plot_rp_3d_new(im, epsilon, plot_path="img/latex/", name=f"turing_example_3drp_new")

# %% line sampling

rp_sum, diag_ind, diag_lengths, diag_dirs, _ = grqa.diag_hs_line_sampling(
    rp, 2, verbose=True, rot=True, return_distribution=False
)


rp_sum, vert_ind, vert_lengths, vert_dirs, _ = grqa.vert_hs_line_sampling(
    rp, 2, verbose=True, rot=True, return_distribution=False
)


# %% diagonal line example

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

cb = ax.imshow(im, origin="lower", cmap="Greys_r", rasterized=True)
ax.set_xlabel("x", fontsize=20, labelpad=10)
ax.set_ylabel("y", fontsize=20, labelpad=15)
ax.tick_params(axis="both", labelsize=16)
ax.set_xticks(np.arange(0, 20, 2))
ax.set_yticks(np.arange(0, 20, 2))
# plt.colorbar(cb, ax=ax, fraction=0.046)

ind_mask = np.zeros(im.shape)

# up diagonal example
# idx = (0, 9)
# lines = np.where(
#    (diag_ind[:, 0] == idx[0])
#    & (diag_ind[:, 1] == idx[1])
#    & (np.sum((diag_dirs == np.array([1, 0, 1, 1])), axis=1) == 4)
#    & (diag_lengths == 5)
# )[0]

# up up example
idx = (0, 9)
lines = np.where(
    (diag_ind[:, 0] == idx[0])
    & (diag_ind[:, 1] == idx[1])
    & (diag_ind[:, 2] == 4)
    & (diag_ind[:, 3] == 1)
    & (np.sum((diag_dirs == np.array([1, 0, 1, 0])), axis=1) == 4)
    & (diag_lengths == 5)
)[0]


for l in [lines[0]]:
    for length in range(diag_lengths[l]):
        ind_mask[
            diag_ind[l, 0] + diag_dirs[l, 0] * length,
            diag_ind[l, 1] + diag_dirs[l, 1] * length,
        ] = 1
        ind_mask[
            diag_ind[l, 2] + diag_dirs[l, 2] * length,
            diag_ind[l, 3] + diag_dirs[l, 3] * length,
        ] = 1

ax.imshow(
    ind_mask, cmap="plasma", vmin=0, vmax=1, alpha=0.7, origin="lower", rasterized=True
)

plt.show()
# plt.savefig("img/latex/turing_example_up_up.png", dpi=300, bbox_inches="tight")
# plt.savefig("img/latex/turing_example_up_up.pdf", dpi=300, bbox_inches="tight")
plt.close()


# %% vertical line example

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

cb = ax.imshow(im, origin="lower", cmap="Greys_r")
ax.set_xlabel("x", fontsize=20, labelpad=10)
ax.set_ylabel("y", fontsize=20, labelpad=15)
ax.tick_params(axis="both", labelsize=16)
ax.set_xticks(np.arange(0, 20, 2))
ax.set_yticks(np.arange(0, 20, 2))
# plt.colorbar(cb, ax=ax, fraction=0.046)

ind_mask = np.zeros(im.shape)

# vert right example
# idx = (0, 1)
# lines = np.where(
#    (vert_ind[:, 0] == idx[0])
#    & (vert_ind[:, 1] == idx[1])
#    & (np.sum((vert_dirs == np.array([0, 1])), axis=1) == 2)
#    & (vert_lengths == 5)
# )[0]

# vert up example
idx = (0, 1)
lines = np.where(
    (vert_ind[:, 0] == idx[0])
    & (vert_ind[:, 1] == idx[1])
    & (np.sum((vert_dirs == np.array([1, 0])), axis=1) == 2)
    & (vert_lengths == 5)
)[0]

for l in [lines[0]]:
    for length in range(vert_lengths[l]):
        ind_mask[vert_ind[l, 0], vert_ind[l, 1]] = 1
        ind_mask[
            vert_ind[l, 2] + vert_dirs[l][0] * length,
            vert_ind[l, 3] + vert_dirs[l, 1] * length,
        ] = 1

ax.imshow(
    ind_mask, cmap="plasma", vmin=0, vmax=1, alpha=0.7, origin="lower", rasterized=True
)

plt.show()
# plt.savefig("img/latex/turing_example_vert_up.png", dpi=300, bbox_inches="tight")
# plt.savefig("img/latex/turing_example_vert_up.pdf", dpi=300, bbox_inches="tight")
plt.close()


# %% patch sampling

rp_sum, diag_ind, diag_lengths, diag_dirs, _ = grqa.diag_hs_patch_sampling(
    rp, 2, verbose=True, rot=True
)
rp_sum, vert_ind, vert_lengths, vert_dirs, _ = grqa.vert_hs_patch_sampling(
    rp, 2, verbose=True, rot=True
)


# %% diagonal patch example

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

cb = ax.imshow(im, origin="lower", cmap="Greys_r")
ax.set_xlabel("x", fontsize=20, labelpad=10)
ax.set_ylabel("y", fontsize=20, labelpad=15)
ax.tick_params(axis="both", labelsize=16)
ax.set_xticks(np.arange(0, 20, 2))
ax.set_yticks(np.arange(0, 20, 2))
# plt.colorbar(cb, ax=ax, fraction=0.046)

ind_mask = np.zeros(im.shape)

# up up example
# idx = (16, 12, 6, 2)
# sfcs = np.where(
#    (diag_ind[:, 0] == idx[0])
#    & (diag_ind[:, 1] == idx[1])
#    & (np.sum((diag_dirs == np.array([1, 1, 1, 1])), axis=1) == 4)
#    & (diag_lengths == 4)
# )[0]

# up down example
idx = (14, 15, 17, 8)
sfcs = np.where(
    (diag_ind[:, 0] == idx[0])
    & (diag_ind[:, 1] == idx[1])
    & (np.sum((diag_dirs == np.array([1, 1, -1, 1])), axis=1) == 4)
    & (diag_lengths == 5)
)[0]

for s in [sfcs[0]]:
    for length1 in range(diag_lengths[s]):
        for length2 in range(diag_lengths[s]):
            ind_mask[
                diag_ind[s, 0] + diag_dirs[s, 0] * length1,
                diag_ind[s, 1] + diag_dirs[s, 1] * length2,
            ] = 1
            ind_mask[
                diag_ind[s, 2] + diag_dirs[s, 2] * length1,
                diag_ind[s, 3] + diag_dirs[s, 3] * length2,
            ] = 1

ax.imshow(
    ind_mask, cmap="plasma", vmin=0, vmax=1, alpha=0.7, origin="lower", rasterized=True
)

# plt.savefig(f"img/latex/turing_example_patch_up_down.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"img/latex/turing_example_patch_up_down.pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# %% vertical patch example

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

cb = ax.imshow(im, origin="lower", cmap="Greys_r")
ax.set_xlabel("x", fontsize=20, labelpad=10)
ax.set_ylabel("y", fontsize=20, labelpad=15)
ax.tick_params(axis="both", labelsize=16)
ax.set_xticks(np.arange(0, 20, 2))
ax.set_yticks(np.arange(0, 20, 2))
# plt.colorbar(cb, ax=ax, fraction=0.046)

ind_mask = np.zeros(im.shape)

# up right example
# idx = (0, 4)
# sfcs = np.where(
#    (vert_ind[:, 0] == idx[0])
#    & (vert_ind[:, 1] == idx[1])
#    & (np.sum((vert_dirs == np.array([1, 1])), axis=1) == 2)
#    & (vert_lengths == 3)
# )[0]

# up left example
idx = (0, 4)
sfcs = np.where(
    (vert_ind[:, 0] == idx[0])
    & (vert_ind[:, 1] == idx[1])
    & (np.sum((vert_dirs == np.array([-1, 1])), axis=1) == 2)
    & (vert_lengths == 3)
)[0]
# the 7th

for s in [sfcs[0]]:
    for length1 in range(vert_lengths[s]):
        for length2 in range(vert_lengths[s]):
            ind_mask[
                vert_ind[s, 0],
                vert_ind[s, 1],
            ] = 1
            ind_mask[
                vert_ind[s, 2] + vert_dirs[s, 0] * length1,
                vert_ind[s, 3] + vert_dirs[s, 1] * length2,
            ] = 1

ax.imshow(
    ind_mask, cmap="plasma", vmin=0, vmax=1, alpha=0.7, origin="lower", rasterized=True
)

# plt.savefig(f"img/latex/turing_example_patch_vert_up_left.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"img/latex/turing_example_patch_vert_up_left.pdf", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# %% 2d rp examples from different time series

for name in ["rand", "sin", "combined", "stag", "stag_2", "regime", "long"]:

    ### random
    if name == "rand":
        rand = np.array([np.random.random() for _ in range(100)])
        ts = (rand - 0.5) * 2

    ### sin
    elif name == "sin":
        sin = np.array([np.sin(i) for i in np.linspace(0, 10 * np.pi, 100)])
        ts = sin

    ### combined
    elif name == "combined":
        sin = np.array([np.sin(i) for i in np.linspace(0, 10 * np.pi, 100)])
        rand = np.array([np.random.random() for _ in range(100)])
        ts = sin + (rand - 0.5)

    ### stagnating
    elif name == "stag":
        sin = np.array([np.sin(i) for i in np.linspace(0, 10 * np.pi, 80)])
        ts = np.hstack(
            [
                sin[:30],
                np.ones(20) * np.sin(np.linspace(0, 10 * np.pi, 80)[30]),
                sin[30:],
            ]
        )

    ### stagnating x 2
    elif name == "stag_2":
        sin = np.array([np.sin(i) for i in np.linspace(0, 10 * np.pi, 70)])
        ts = np.hstack(
            [
                sin[:20],
                np.ones(20) * np.sin(np.linspace(0, 10 * np.pi, 70)[20]),
                sin[20:42],
                np.ones(10) * np.sin(np.linspace(0, 10 * np.pi, 70)[42]),
                sin[42:],
            ]
        )

    ### regime shift
    elif name == "regime":
        sin1 = np.array([np.sin(i) for i in np.linspace(0, 3 * np.pi, 60)])
        sin2 = np.array([np.sin(i) for i in np.linspace(0, 9 * np.pi, 40)]) + 1
        ts = np.hstack([sin1, sin2])

    ### long characteristic time
    elif name == "long":
        sin = np.array([np.sin(i) for i in np.linspace(0, 1.7 * np.pi, 100)])
        ts = sin

    dm = squareform(pdist(np.c_[ts, ts], metric="euclidean")) / np.sqrt(2)
    epsilon = np.quantile(dm.flatten(), 0.2)
    rp = np.zeros(dm.shape)
    rp[dm <= epsilon] = 1
    rp = rp.astype(np.int32)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

    ax1.plot(ts, c="darkblue", lw=2)
    ax1.set_ylim(-2.1, 2.1)
    ax1.set_title("Time Series", fontsize=28, fontweight="bold", pad=30)
    ax1.set_xlabel("time", fontsize=22)
    ax1.set_ylabel("value", fontsize=22)
    ax1.set_xticks([0, 20, 40, 60, 80])
    ax1.tick_params(axis="both", labelsize=20)
    ax1.grid()

    im2 = ax2.imshow(
        dm,
        origin="lower",
        interpolation="none",
        cmap="OrRd",
        rasterized=True,
        vmin=0,
        vmax=3,
    )
    ax2.set_title("Distance Matrix", fontsize=28, fontweight="bold", pad=30)
    ax2.set_xlabel("time", fontsize=22)
    ax2.set_xticks([0, 20, 40, 60, 80])
    ax2.set_ylabel("time", fontsize=22)
    ax2.tick_params(axis="both", labelsize=20)
    cbar = plt.colorbar(im2, ax=ax2)
    cbar.set_label("distance", fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    cmap = cm.get_cmap("Greys_r", 2)
    im3 = ax3.imshow(
        rp, origin="lower", interpolation="none", cmap=cmap, rasterized=True
    )
    ax3.set_title("Recurrence Plot", fontsize=28, fontweight="bold", pad=30)
    ax3.set_xlabel("time", fontsize=22)
    ax3.set_xticks([0, 20, 40, 60, 80])
    ax3.set_ylabel("time", fontsize=22)
    ax3.tick_params(axis="both", labelsize=20)
    cbar = plt.colorbar(im3, ax=ax3, ticks=[0.25, 0.75])
    cbar.set_label("recurrence", fontsize=22, labelpad=10)
    cbar.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 20})

    plt.subplots_adjust(wspace=0.3)

    plt.savefig(f"img/latex/exp_{name}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"img/latex/exp_{name}.pdf", bbox_inches="tight", dpi=300)
    #plt.show()
    plt.close()


# %% lacunarity example 4d

img = utils.get_periodic_test_image(20, n=1.5)

(
    mass_distribution,
    mass_distribution_reverse,
    mass_distribution_z2,
    mass_distribution_reverse_z2,
    nr_boxes,
) = grqa.mass_distribution_fast(img, max_box_size=20)

lac = grqa.lacunarity(
    mass_distribution,
    mass_distribution_reverse,
    mass_distribution_z2,
    mass_distribution_reverse_z2,
    nr_boxes,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(img)
ax2.loglog(range(len(nr_boxes)), lac)

plt.show()
plt.close()


# %% lacunarity example: lac=1 and lac=0, very small

### lac = 1
im = np.zeros((6, 6))
im[:2, :2] = 1
for i in range(len(im)):
    im[i, i] = 1

### lac = 0
#im = 0
#im = np.zeros((6, 6))
#im[::2, ::2] = 1

for i in range(len(im)):
    im[i, i] = 1

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.imshow(im, interpolation="None", origin="lower", cmap="Greys_r")

ax.set_xlabel("time", fontsize=26, labelpad=10)
ax.set_ylabel("time", fontsize=26, labelpad=10)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.tick_params(axis="both", labelsize=22)

plt.show()
#plt.savefig("img/latex/lac1_exp.png", bbox_inches="tight", dpi=300)
plt.close()

box_sizes, lac = grqa.mass_distribution_2d(im, 3)
print(box_sizes)
print(lac)


# %% lacunarity example: lac=1 and lac=0, for appendix

# rp = np.ones((50, 50))
rp = np.zeros((6, 6))
rp[:2, :2] = 1
# rp[1, 1] = 0
# rp[::2, ::2] = 1
# rp[0, 0] = 0
# rp[0:1, 2:4] = 1
# rp[1, 2] = 1


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
cb = ax.imshow(rp, origin="lower", cmap="Greys_r", rasterized=True)
ax.set_ylabel("y", fontsize=14)
ax.set_xlabel("x", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
cbar = plt.colorbar(cb, ax=ax, fraction=0.046)
cbar.set_label("value", fontsize=14)
cbar.ax.tick_params(labelsize=12)
plt.show()
# plt.savefig("img/latex/lac_anhang_1.png", dpi=300, bbox_inches="tight")
# plt.savefig("img/latex/lac_anhang_1.pdf", dpi=300, bbox_inches="tight")
plt.close()

box_sizes, lac = grqa.mass_distribution_2d(rp, 2)

print(box_sizes)
print(lac)


# %% lac example: timeseries

fig, ax = plt.subplots(3, 1, figsize=(4, 10), gridspec_kw={"height_ratios": [3, 5, 3]})

### periodic signal
# RR = 0.2
# x = np.arange(200)
# y = np.sin(1 * x / (2 * np.pi)) + np.sin(2 * x / (2 * np.pi))
# x = np.arange(y.shape[0])

### regime shift
RR = 0.1
x = np.arange(100)
y = np.hstack(
    [
        np.random.random(20),
        2 * np.random.random(20) + 10,
        3 * np.random.random(20) + 20,
        4 * np.random.random(20) + 30,
        5 * np.random.random(20) + 40,
    ]
)

dm = squareform(pdist(np.c_[y, y], metric="euclidean")) / np.sqrt(2)
epsilon = np.quantile(dm.flatten(), RR)
rp = np.zeros(dm.shape)
rp[dm <= epsilon] = 1
rp = rp.astype(np.int32)

box_sizes, lac = grqa.mass_distribution_2d(rp, 50)

ax[0].plot(x, y, c="navy", lw=2)
ax[0].set_title("Time Series", fontsize=20, fontweight="bold", pad=20)
ax[0].set_xlabel("time", fontsize=14)
ax[0].set_ylabel("value", fontsize=14, labelpad=10)
ax[0].tick_params(axis="both", labelsize=12)
ax[0].grid()

cmap = cm.get_cmap("Greys_r", 2)
im = ax[1].imshow(rp, interpolation="None", origin="lower", cmap=cmap, rasterized=True)
ax[1].set_aspect("equal", "box")
ax[1].set_title("Recurrence Plot", fontsize=20, fontweight="bold", pad=20)
ax[1].set_xlabel("time", fontsize=14)
ax[1].set_ylabel("time", fontsize=14)
ax[1].set_xticks(np.arange(0, rp.shape[1], 40))
ax[1].set_yticks(np.arange(0, rp.shape[0], 40))
ax[1].tick_params(axis="both", labelsize=12)
cbar = plt.colorbar(im, ax=ax[1], label="recurrence", ticks=[0.25, 0.75])
cbar.set_label("recurrence", fontsize=14, labelpad=10)
cbar.ax.set_yticklabels(["0", "1"], fontdict={"fontsize": 14})

ax[2].loglog(box_sizes, lac, c="darkorange", lw=2)
ax[2].set_title("Recurrence Lacunarity", fontsize=20, fontweight="bold", pad=20)
ax[2].set_xlabel("box size", fontsize=14)
ax[2].set_ylabel("lacunarity", fontsize=14, labelpad=10)
ax[2].tick_params(axis="both", labelsize=12)
ax[2].grid()

plt.subplots_adjust(hspace=0.7)

plt.show()
# plt.savefig("img/latex/lac_ts_exp_peak.png", bbox_inches="tight", dpi=300)
# plt.savefig("img/latex/lac_ts_exp_peak.pdf", bbox_inches="tight", dpi=300)
plt.close()


# %% checkerboard examples

# im = np.zeros((20, 20))
# im[::2, :] = 1
# im[::2, ::2] = 2
# im[1::2, 1::2] = 3

im = np.kron([[1, 0] * 2, [0, 1] * 2] * 2, np.ones((4, 4)))

# im = np.random.choice([0, 1, 2, 3, 4, 5], size=(20, 20))
# im[0, :] = 1
# im[1, :] = 2
# im[2, :] = 3

# im = utils.get_periodic_test_image(50, 1, 20)

dm = grqa.distance_matrix(im)
epsilon = grqa.recurrence_threshold(dm, 0.2)
rp = grqa.recurrence_plot(dm, 0)

utils.plot_rp_3d_from_rp(rp, plot_path="img/zzz_old/", name=f"det_lam_example")

m = plt.imshow(im, origin="lower")
plt.colorbar(m)
plt.show()
plt.close()

(
    rp_sum,
    lengths,
    diag_frequ,
    vert_frequ,
    diag_dir_fac,
    vert_dir_fac,
    rr,
) = grqa.hs_line_sampling_fast(im, epsilon=0, rot=True)

det, l = grqa.determinism_and_avg_diag_hs_size(
    diag_frequ, lengths, rp_sum, diag_dir_fac
)
lam, ts = grqa.laminarity_and_trapping_size(vert_frequ, lengths, rp_sum, vert_dir_fac)

print("det:", det)
print("l:", l)
print("lam:", lam)
print("ts:", ts)
