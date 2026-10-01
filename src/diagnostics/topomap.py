"""Scalp maps from the ZuCo channel locations (no MNE needed)."""

import numpy as np

from ..neurolm.channel_mapping import load_zuco_chanlocs


def channel_positions_2d(drop_reference=True):
    """Azimuthal-equidistant projection of the ZuCo electrodes, nose up."""
    _, rows = load_zuco_chanlocs()
    if drop_reference:
        rows = rows[:-1]
    xyz = np.array([[float(r["X"]), float(r["Y"]), float(r["Z"])] for r in rows])
    unit = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    # EEGLAB frame: +X nose, +Y left ear, +Z up.
    polar = np.arccos(np.clip(unit[:, 2], -1, 1))
    azimuth = np.arctan2(unit[:, 1], unit[:, 0])
    return np.stack([-polar * np.sin(azimuth), polar * np.cos(azimuth)], axis=1), [r["labels"] for r in rows]


def plot_topomaps(values, titles, path, suptitle=None, cmap="RdBu_r"):
    """``values``: ``[n_maps, n_channels]``; one interpolated scalp map per row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    positions, _ = channel_positions_2d()
    limit = float(np.nanmax(np.abs(values))) or 1.0
    grid = np.linspace(-1.9, 1.9, 120)
    gx, gy = np.meshgrid(grid, grid)
    columns = min(4, len(values))
    rows = int(np.ceil(len(values) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows), squeeze=False)
    image = None
    for ax, row, title in zip(axes.flat, values, titles):
        surface = griddata(positions, row, (gx, gy), method="cubic")
        surface[np.hypot(gx, gy) > np.abs(positions).max() * 1.05] = np.nan
        image = ax.imshow(surface, origin="lower", extent=(grid[0], grid[-1], grid[0], grid[-1]),
                          cmap=cmap, vmin=-limit, vmax=limit)
        ax.scatter(positions[:, 0], positions[:, 1], s=3, c="k")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(values):]:
        ax.axis("off")
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6)
    if suptitle:
        figure.suptitle(suptitle)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
