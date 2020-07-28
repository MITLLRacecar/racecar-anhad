# https://github.com/KojiKobayashi/iterative_closest_point_2d

import cv2 as cv
import numpy as np

# import sys

import time
from numpy.random import *

# import matplotlib.pyplot as plt
import math

show_animation = False


def del_miss(indices, dist, max_dist, th_rate=0.8):
    th_dist = max_dist * th_rate
    return np.array([indices[0][np.where(dist.T[0] < th_dist)]])


def is_converge(Tr, scale):
    delta_angle = 0.0001
    delta_scale = scale * 0.0001

    min_cos = 1 - delta_angle
    max_cos = 1 + delta_angle
    min_sin = -delta_angle
    max_sin = delta_angle
    min_move = -delta_scale
    max_move = delta_scale

    return (
        min_cos < Tr[0, 0]
        and Tr[0, 0] < max_cos
        and min_cos < Tr[1, 1]
        and Tr[1, 1] < max_cos
        and min_sin < -Tr[1, 0]
        and -Tr[1, 0] < max_sin
        and min_sin < Tr[0, 1]
        and Tr[0, 1] < max_sin
        and min_move < Tr[0, 2]
        and Tr[0, 2] < max_move
        and min_move < Tr[1, 2]
        and Tr[1, 2] < max_move
    )


def icp(d1, d2, max_iterate=200):
    t = time.time()
    src = np.array([d1.T], copy=True).astype(np.float32)
    dst = np.array([d2.T], copy=True).astype(np.float32)

    knn = cv.ml.KNearest_create()  # cv.KNearest()
    responses = np.arange(len(d2[0]), dtype=np.float32)
    knn.train(src[0], cv.ml.ROW_SAMPLE, responses)

    # Tr = np.array([[np.cos(0), -np.sin(0), 0], [np.sin(0), np.cos(0), 0], [0, 0, 1]])
    Tr = np.identity(3)

    dst = cv.transform(dst, Tr[0:2])
    max_dist = 10000  # sys.maxint

    scale_x = np.max(d1[0]) - np.min(d1[0])
    scale_y = np.max(d1[1]) - np.min(d1[1])
    scale = max(scale_x, scale_y)

    for i in range(max_iterate):
        ret, results, neighbours, dist = knn.findNearest(dst[0], 1)

        indices = results.astype(np.int32).T
        indices = del_miss(indices, dist, max_dist)

        T, _ = cv.estimateAffine2D(dst[0, indices], src[0, indices])
        # , True) estimateAffinePartial2D is faster, less accurate

        max_dist = np.max(dist)
        dst = cv.transform(dst, T)
        Tr = np.dot(np.vstack((T, [0, 0, 1])), Tr)

        # if show_animation:  # pragma: no cover
        #    plt.cla()
        #    # for stopping simulation with the esc key.
        #    plt.gcf().canvas.mpl_connect(
        #        "key_release_event",
        #        lambda event: [exit(0) if event.key == "escape" else None],
        #    )
        #    plt.plot(d1[0, :], d1[1, :], ".r")
        #    plt.plot(d2[0, :], d2[1, :], ".b")
        #    plt.plot(0.0, 0.0, "xr")
        #    plt.axis("equal")
        #    plt.pause(1)

        if is_converge(T, scale):
            break
    else:
        print("ICP failed to converge!")

    print("ICP Took: ", time.time() - t)
    return Tr[0:2]


#    th = np.pi / 8
#    move = np.array([[0.30], [0.5]])
#    rnd_scale = 0.03
#    x1 = np.linspace(0, 1.1, point_count)
#    y1 = np.sin(x1 * np.pi)
#    d1 = np.array([x1, y1])

#   rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
#   rand = np.random.rand(2, point_count) * rnd_scale
#   d2 = np.dot(rot, d1) + move
#    d2 = np.add(d2, rand)

#    plt.plot(d1[0], d1[1])
#    plt.plot(d2[0], d2[1])
#    plt.show()
#    ret = icp(d1, d2)

#    plt.plot(d1[0], d1[1])
#    dst = np.array([d2.T], copy=True).astype(np.float32)
#    dst = cv.transform(dst, ret)
#    plt.plot(dst[0].T[0], dst[0].T[1])
#    plt.show()

# print(ret[0][0] * ret[0][0] + ret[0][1] * ret[0][1])
# print(np.arccos(ret[0][0]) / 2 / np.pi * 360)
# print(np.arcsin(ret[0][1]) / 2 / np.pi * 360)#

#    print(ret)


def main():
    print(__file__ + " start")

    # simulation parameters
    nPoint = 1000
    fieldLength = 300.0
    motion = [0.5, 2.0, np.deg2rad(-10.0)]  # movement [x[m],y[m],yaw[deg]]

    nsim = 3  # number of simulation

    for _ in range(nsim):
        t = time.time()

        # previous points
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py))

        # print(previous_points)

        # current points
        cx = [
            math.cos(motion[2]) * x - math.sin(motion[2]) * y + motion[0]
            for (x, y) in zip(px, py)
        ]
        cy = [
            math.sin(motion[2]) * x + math.cos(motion[2]) * y + motion[1]
            for (x, y) in zip(px, py)
        ]
        current_points = np.vstack((cx, cy))

        R = icp(previous_points, current_points)
        print("R:", R)
        # print("T:", T)
        print(time.time() - t)


if __name__ == "__main__":
    main()
