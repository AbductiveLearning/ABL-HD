import pickle
import os
import numpy as np
import math
import functools
import multiprocessing as mp

from tqdm import tqdm
from shapely.geometry import Polygon
from shapely.geometry import Point


def y_small2large(a, b):
    if a[0][1] < b[0][1]:
        return -1
    return 1


def cal_angle(v1):
    theta = np.arccos(min(1, v1[0] / (np.linalg.norm(v1) + 10e-8)))
    return 2 * math.pi - theta if v1[1] < 0 else theta


def clockwise_sort(points):
    # returns 4x2 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] ndarray
    v1, v2, v3, v4 = points
    center = (v1 + v2 + v3 + v4) / 4
    theta = np.array(
        [
            cal_angle(v1 - center),
            cal_angle(v2 - center),
            cal_angle(v3 - center),
            cal_angle(v4 - center),
        ]
    )
    index = np.argsort(theta)
    return np.roll(np.array([v1, v2, v3, v4])[index, :], 2, axis=0)


def match(one_img_text_line, character_bboxes, extra_info):
    column_level_character_bboxes = []
    for i in range(len(one_img_text_line)):
        tmp = []
        poly1 = Polygon(one_img_text_line[i])
        poly1 = poly1.buffer(0.01)
        for j in range(len(character_bboxes)):
            poly2 = Polygon(character_bboxes[j]).convex_hull
            if poly2.area < 10:
                continue
            center = Point((character_bboxes[j][0] + character_bboxes[j][2]) / 2)
            if poly1.intersection(poly2).area / poly2.area > 0.25 and poly1.contains(
                center
            ):
                tmp.append(character_bboxes[j])
        tmp = sorted(tmp, key=functools.cmp_to_key(y_small2large))
        tmp.append(extra_info[i])

        column_level_character_bboxes.append(tmp)
    return column_level_character_bboxes


def fast_match(one_img_text_line, character_bboxes, extra_info):
    character_bboxes = sorted(character_bboxes, key=lambda i: i[0][0])
    column_level_character_bboxes = []
    new_polys = []
    for i in range(len(one_img_text_line)):
        tmp = []

        x_min, y_min = np.min(one_img_text_line[i], axis=0)
        st = next(
            (indx for indx, box in enumerate(character_bboxes) if box[1][0] >= x_min),
            None,
        )
        if st is not None:
            poly1 = Polygon(one_img_text_line[i]).buffer(0.01)
            x_max, y_max = np.max(one_img_text_line[i], axis=0)
            for j in range(st, len(character_bboxes)):
                if character_bboxes[j][0][0] >= x_max:
                    break
                if (
                    character_bboxes[j][1][0] <= x_min
                    or character_bboxes[j][0][1] >= y_max
                    or character_bboxes[j][2][1] <= y_min
                ):
                    continue

                center = (character_bboxes[j][0] + character_bboxes[j][2]) / 2
                if (
                    center[0] < x_min
                    or center[0] > x_max
                    or center[1] < y_min
                    or center[1] > y_max
                ):
                    continue

                center = Point((character_bboxes[j][0] + character_bboxes[j][2]) / 2)
                if poly1.contains(center):
                    tmp.append(character_bboxes[j])
            if len(tmp) > 0:
                tmp = sorted(tmp, key=lambda item: item[0][1])
                tmp.append(extra_info[i])
                column_level_character_bboxes.append(tmp)
                new_polys.append(one_img_text_line[i])
    return new_polys, column_level_character_bboxes


def has_column_gt(word_bbox_dir, character_bbox_dir):
    img_names = os.listdir(word_bbox_dir)
    img_names = [img_name.split(".")[0] for img_name in img_names]

    res = {}

    for img_name in tqdm(img_names):
        one_img_text_line = []
        words = []
        with open(word_bbox_dir + img_name + ".txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
                if len(box_info) == 9:
                    box_points = [int(box_info[i]) for i in range(8)]
                    box_points = np.array(box_points, np.float32).reshape(4, 2)
                    box_points = clockwise_sort(box_points)
                    word = box_info[8:][0]
                else:
                    poly_points = np.array(
                        [int(box_info[i]) for i in range(32)], np.float32
                    ).reshape(16, 2)
                    box_points = poly_points
                    word = box_info[32:][0]

                box_points = np.array(box_points, np.float32).reshape(-1, 2)
                one_img_text_line.append(box_points)
                words.append(word)

        character_bboxes = []
        with open(
            character_bbox_dir + img_name.replace("gt_", "") + ".txt",
            "r",
            encoding="utf-8",
        ) as f:
            for line in f.readlines():
                box_info = line.strip().encode("utf-8").decode("utf-8-sig").split(",")
                box_points = [int(box_info[i]) for i in range(8)]
                box_points = np.array(box_points, np.float32).reshape(4, 2)
                box_points = clockwise_sort(box_points)
                character_bboxes.append(box_points)

        column_level_character_bboxes = match(
            one_img_text_line, character_bboxes, words, img_name
        )
        res[img_name] = column_level_character_bboxes

    with open(character_bbox_dir + "predict_character_bbox.pickle", "wb") as f:
        pickle.dump(res, f)
