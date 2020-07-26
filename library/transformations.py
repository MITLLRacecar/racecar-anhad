#    Y+
# X- car X+
#    Y-

# icp takes points as [[x, x, x, x], [y, y, y, y]]

import numpy as np

VIS_RADIUS = 0
width = 0
num_samples = 0


def p2c(r, t):
    """polar r (float), theta (float) to cartesian x (float), y (float)"""
    if np.isfinite(r) and np.isfinite(t):
        return r * np.sin(t), r * np.cos(t)  # x, y
    else:
        return 0, 0  # return 0, 0


def c2p(x, y):
    """cartesian x (float), y (float) to polar r (float), theta (float)"""
    if np.isfinite(x) and np.isfinite(y):
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)
    else:
        return 0, 0  # return 0, 0


Vp2c = np.vectorize(p2c, [np.float32, np.float32])
Vc2p = np.vectorize(c2p, [np.float32, np.float32])


def lidar2Polar(scan, start=0, end=360):
    """converts lidar scan to [[r, r, r], [t, t, t]]"""
    r = 360.0 / num_samples
    scan_r = np.take(scan, np.arange(start / r, end / r).astype(np.int), mode="wrap")
    # scan_r[abs(scan_r - np.mean(scan_r)) > 2 * np.std(scan_r)] = 0 #remove outliers
    scan_t = np.radians(np.arange(start, end, step=r))

    # valid_idx = np.argwhere(np.logical_and(scan_r > 0, scan_r <= VIS_RADIUS)).flatten()
    valid_idx = np.argwhere(scan_r > 0).flatten()

    return np.array([np.take(scan_r, valid_idx), np.take(scan_t, valid_idx)])


def camera2Polar(points, depths):  # t, r
    """transform points in color image space to polar space, [[r, r, r], [t, t, t]]"""
    # fov 45 degrees
    angles = np.radians(
        (points[:, 1] - width // 2) * 69.4 / width  # 74
    )  # convert from x to angle

    valid_idx = np.argwhere(np.logical_and(depths > 0, depths <= VIS_RADIUS)).flatten()

    return np.array([np.take(depths, valid_idx), np.take(angles, valid_idx)])


def polar2TopDown(r, t=None):
    """convert polar [r, r, r], [t, t, t] or  [[r, r, r], [t, t, t]] to cartesian [[x, x, x], [y, y, y]], centered on 0"""
    if t is None:
        return np.array(Vp2c(r[0], r[1]))
    else:
        return np.array(Vp2c(r, t))


def topDown2Vis(points, center):
    """transform points in top down space [[x, x, x], [y, y, y]] to visualization space [[y, x], [y, x], [y, x]], centered on center [x, y]"""
    if points is None or points.size == 0:
        return None

    valid_idx = np.argwhere((np.absolute(points) < VIS_RADIUS).all(axis=0)).flatten()
    return (
        tuple(center[1] - np.take(points[1], valid_idx).astype(int)),
        tuple(np.take(points[0], valid_idx).astype(int) + center[0]),
    )


def vis2TopDown(points, center):
    """invert topDown2Vis: [[y, x], [y, x], [y, x]] centered on center [x, y] to [[x, x, x], [y, y, y]]"""
    if points is None or points.size == 0:
        return None

    i = points.astype(float)
    return np.array([i[:, 1] - center[1], center[0] - i[:, 0]])
