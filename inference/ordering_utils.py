import cv2 as cv
import numpy as np
from scipy.signal import find_peaks


def assign_group(parsed_bboxes, prominence_factor=0.71):
    sorted_ranges = sorted(
        [[item["y_start"], item["y_end"]] for item in parsed_bboxes.values()]
    )
    global_min = min([i[0] for i in sorted_ranges])
    global_max = max([i[1] for i in sorted_ranges])

    # Find clear gaps
    non_overlapping_ranges = []
    for begin, end in sorted(sorted_ranges):
        if non_overlapping_ranges and non_overlapping_ranges[-1][1] >= begin - 1:
            non_overlapping_ranges[-1][1] = max(non_overlapping_ranges[-1][1], end)
        else:
            non_overlapping_ranges.append([begin, end])
    peaks1 = [min(0, global_min), *[i[1] - global_min for i in non_overlapping_ranges]]

    # Find fuzzy gaps
    range_size = global_max - global_min
    counts = np.zeros(range_size, dtype=np.int16)
    for begin, end in sorted(sorted_ranges):
        counts = np.add(
            counts,
            [0] * (begin - global_min) + [1] * (end - begin) + [0] * (global_max - end),
        )

    average = np.mean([i for i in counts if i > 0])
    peaks2, _ = find_peaks(-counts, prominence=average * prominence_factor)

    # Combine them
    peaks = sorted(set([*peaks1, *peaks2]))

    # Assign group numbers according to gaps
    for index in parsed_bboxes.keys():
        y_start = parsed_bboxes[index]["y_start"]
        y_end = parsed_bboxes[index]["y_end"]
        mean = (y_start + y_end) / 2 - global_min

        for group_index in range(len(peaks) - 1):
            if mean >= peaks[group_index] and mean <= peaks[group_index + 1]:
                parsed_bboxes[index]["group"] = group_index
                break

        if parsed_bboxes[index].get("group") is None:
            ## Should not reach here!
            # print(mean)
            # assert 0
            parsed_bboxes[index]["group"] = 0

    return len(peaks) - 1


def predict_order(bboxes, image_shape, prominence_factor=0.74):
    """
    input:
        bboxes: e.g., [1286, 59, 1326, 59, 1331, 851, 1290, 851]
        image_shape: e.g., (1080, 1920, *)
    output:
        a permutation of range(len(bboxes)), e.g., [0, 1, 3, 2]
    """

    if not len(bboxes):
        return []

    parsed_bboxes = {}
    sum_v1 = np.array([0, 0])
    sum_v2 = np.array([0, 0])

    for index, parsed_points in enumerate(bboxes):

        # Get estimated upper-left, upper-right, lower-left, lower-right position of bbox
        temp = sorted(parsed_points, key=lambda i: i[0] + i[1])
        upper_left_point = temp[0]
        lower_right_point = temp[-1]

        temp = sorted(parsed_points, key=lambda i: i[0] - i[1])
        lower_left_point = temp[0]
        upper_right_point = temp[-1]

        v1 = np.subtract(lower_right_point, upper_right_point) + np.subtract(
            lower_left_point, upper_left_point
        )
        v2 = np.subtract(upper_left_point, upper_right_point) + np.subtract(
            lower_left_point, lower_right_point
        )

        sum_v1 += v1
        sum_v2 += v2

        parsed_bboxes[index] = {
            "upper_left_point": upper_left_point,
            "lower_right_point": lower_right_point,
            "lower_left_point": lower_left_point,
            "upper_right_point": upper_right_point,
        }

    norm_v1 = np.linalg.norm(sum_v1)
    norm_v2 = np.linalg.norm(sum_v2)

    if norm_v1 > norm_v2:
        orientation_vec = sum_v1 / norm_v1
    else:
        orientation_vec = sum_v2 / norm_v2
    vertical_vec = np.array([orientation_vec[1], -orientation_vec[0]])

    for index, value in parsed_bboxes.items():
        upper_left_point = value["upper_left_point"]
        lower_right_point = value["lower_right_point"]
        lower_left_point = value["lower_left_point"]
        upper_right_point = value["upper_right_point"]

        y_start = np.dot(np.add(upper_left_point, upper_right_point), orientation_vec)
        y_end = np.dot(np.add(lower_left_point, lower_right_point), orientation_vec)

        x_start = np.dot(np.add(upper_left_point, lower_left_point), vertical_vec)
        x_end = np.dot(np.add(upper_right_point, lower_right_point), vertical_vec)

        parsed_bboxes[index] = {
            "y_start": round(y_start),
            "x_start": round(x_start),
            "y_end": round(y_end),
            "x_end": round(x_end),
        }

    # Assign group numbers for bboxes
    max_group_index = assign_group(parsed_bboxes, prominence_factor=prominence_factor)
    result = []

    for group_index in range(max_group_index + 1):
        sub_group_codes = {
            k: v for k, v in parsed_bboxes.items() if v["group"] == group_index
        }

        ordered_indexes = sorted(
            [k for k in sub_group_codes.keys()],
            key=lambda k: sub_group_codes[k]["y_start"],
        )

        line_finish_x = {}
        while len(ordered_indexes):
            target = ordered_indexes[0]
            line_indexes = [target]

            left_x = sub_group_codes[target]["x_start"]
            right_x = sub_group_codes[target]["x_end"]

            tolerance = image_shape[0] * 0.02

            for index in ordered_indexes[1:]:

                x_start = sub_group_codes[index]["x_start"]
                x_end = sub_group_codes[index]["x_end"]

                if x_start >= left_x - tolerance and x_end <= right_x + tolerance:
                    # rougly covered
                    left_x = min(left_x, x_start)
                    right_x = max(right_x, x_end)
                    line_indexes.append(index)

            for index in line_indexes:
                sub_group_codes[index]["line"] = len(line_finish_x)
                ordered_indexes.remove(index)

            line_finish_x[len(line_finish_x)] = right_x

        ordered_lineno = sorted(
            line_finish_x.keys(), key=lambda i: line_finish_x[i], reverse=True
        )

        for lineno in ordered_lineno:
            indexes_within_line = [
                k for k, v in sub_group_codes.items() if v["line"] == lineno
            ]
            ordered_indexes = sorted(
                indexes_within_line, key=lambda k: sub_group_codes[k]["y_start"]
            )
            result.extend(ordered_indexes)

    # assert set(result) == set(range(len(bboxes)))

    return result


def parse_points(pts):
    xs = pts[::2]
    ys = pts[1::2]
    parsed_points = [(x, y) for x, y in zip(xs, ys)]
    return parsed_points
