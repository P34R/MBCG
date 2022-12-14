import json
import math
import os

import cv2
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot


def task1_NurbsSpline(points, degree):
    data_points = np.array(points, dtype=np.float32)
    p = degree
    n = len(data_points) - 1
    m = n + p + 2
    k = p + 1

    d_vector = [vector_distance(data_points[i + 1], data_points[i]) for i in range(n)]
    # print(d_vector, n)
    d = sum(d_vector)
    # print(d)
    th_vector = np.zeros((n + 1, 1), dtype=np.float32)
    sum_d = 0
    for i in range(n + 1):
        th_vector[i] = sum_d / d
        if i != n:
            sum_d += d_vector[i]
    # print("tvec", th_vector)
    t_vector = np.zeros((m, 1), dtype=np.float32)

    for i in range(m):
        if i < k:
            t_vector[i] = 0.
        elif i >= m - k:
            t_vector[i] = 1.
        else:
            j = i - p
            for l in range(j, j + p):
                t_vector[i] += th_vector[l]
            t_vector[i] *= 1 / p
    # print("tv", t_vector)
    # Wi = (1,1) for every point
    sum_N = 0.0
    # print("T vect", len(t_vector))
    # print("Th vect", len(th_vector))
    q_matr = np.zeros((n + 1, n + 1), dtype=np.float32)
    # print(q_matr)
    for i in range(n + 1):
        sum_N += getN(i, p, th_vector[i], t_vector)
    for i in range(n + 1):
        for j in range(n + 1):
            q_matr[i][j] = getN(j, p, th_vector[i], t_vector)
            # q_matr[i][j] /= sum_N
    q_matr[n][n] = 1  # / sum_N

    P_i = np.linalg.solve(q_matr, points)
    # print(P_i)
    # for i in range(n+1):
    return P_i, th_vector, t_vector


def getN(i, p, t, t_vector):
    if p == 0:
        if (t_vector[i] == t_vector[i + 1] and t_vector[i] <= t) or (t_vector[i] <= t < t_vector[i + 1]):
            return 1
        else:
            return 0
    else:
        left = right = 0
        if t_vector[i + p] != t_vector[i]:
            left = (t - t_vector[i]) / (t_vector[i + p] - t_vector[i])
            left = left * getN(i, p - 1, t, t_vector)
        if t_vector[i + p + 1] != t_vector[i + 1]:
            right = (t_vector[i + p + 1] - t) / (t_vector[i + p + 1] - t_vector[i + 1])
            right = right * getN(i + 1, p - 1, t, t_vector)
        return left + right


def vector_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def draw_points(img, points, color, radius):
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)


def Pt(pi, t, t_v, p):
    sum = pi[0] - pi[0]
    for i in range(len(pi)):
        sum += getN(i, p, t, t_v) * pi[i]
    return sum


def draw(img, pi, th, t, p):
    m = len(th)

    for i in range(m - 1):
        tk = np.linspace(th[i], th[i + 1], 20)
        for j in range(len(tk) - 1):
            start = Pt(pi, tk[j], t, p)
            end = Pt(pi, tk[j + 1], t, p)
            if i == m - 2 and j == len(tk) - 2:
                end = pi[m - 1]  # idk why, but last point is 0,0

            cv2.line(img, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (123, 123, 0), 3)
            cv2.imshow("nurbs", img)


def task2_Bezie(points, indices, n,m):
    uv = np.linspace(0, 1, 51) # surface "beauty"

    p_uv = []
    for u in uv:
        for v in uv:
            point = np.zeros(3)
            for i in range(n):
                for j in range(m):
                    point += float(bernstein(n - 1, i, u)) * float(bernstein(m - 1, j, v)) * points[
                        indices.index([i, j])]
            p_uv.append(point)

    return p_uv


def binom(n, i):
    return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))


def bernstein(n, i, u):
    return binom(n, i) * (u ** i) * ((1 - u) ** (n - i))


if __name__ == '__main__':
    file = open(os.path.join(os.getcwd(), '16.json'))
    data=json.load(file)
    points_task1 = data["curve"]

    # print(points)
    pi, th, t = task1_NurbsSpline(points_task1, 3)

    img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
    draw_points(img, points_task1, (0, 0, 255), 5)
    draw_points(img, pi, (0, 255, 255), 5)
    cv2.imshow("points", img)
    draw(img, pi, th, t, 3)

    surf = data["surface"]
    points_task2 = np.array(surf["points"])
    indices = surf["indices"]  # 13x13 always
    grid = surf["gridSize"]
    p = np.array(task2_Bezie(points_task2, indices, grid[0],grid[1]))
    ax = mpl.pyplot.figure().add_subplot(projection='3d')
    triangulation = mpl.tri.Triangulation(p[:, 0], p[:, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_trisurf(triangulation, p[:, 2])
    ax.view_init(70,120)
    mpl.pyplot.show()
    cv2.waitKey(0)