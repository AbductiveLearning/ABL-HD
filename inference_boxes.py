import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import torch
from torch.autograd import Variable

from craft_utils import getDetBoxes, adjustResultCoordinates, adjustResultCoordinates4character
import imgproc
import math
import xml.etree.ElementTree as elemTree


#-------------------------------------------------------------------------------------------------------------------#
def rotatePoint(xc, yc, xp, yp, theta):
    xoff = xp - xc
    yoff = yp - yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    # pRes = (xc + pResx, yc + pResy)
    return int(xc + pResx), int(yc + pResy)

def addRotatedShape(cx, cy, w, h, angle):
    p0x, p0y = rotatePoint(cx, cy, cx - w / 2, cy - h / 2, -angle)
    p1x, p1y = rotatePoint(cx, cy, cx + w / 2, cy - h / 2, -angle)
    p2x, p2y = rotatePoint(cx, cy, cx + w / 2, cy + h / 2, -angle)
    p3x, p3y = rotatePoint(cx, cy, cx - w / 2, cy + h / 2, -angle)

    points = [[p0x, p0y], [p1x, p1y], [p2x, p2y], [p3x, p3y]]

    return points

def xml_parsing(xml):
    tree = elemTree.parse(xml)

    annotations = []  # Initialize the list to store labels
    iter_element = tree.iter(tag="object")

    for element in iter_element:
        annotation = {}  # Initialize the dict to store labels

        annotation['name'] = element.find("name").text  # Save the name tag value

        box_coords = element.iter(tag="robndbox")

        for box_coord in box_coords:
            cx = float(box_coord.find("cx").text)
            cy = float(box_coord.find("cy").text)
            w = float(box_coord.find("w").text)
            h = float(box_coord.find("h").text)
            angle = float(box_coord.find("angle").text)

            convertcoodi = addRotatedShape(cx, cy, w, h, angle)

            annotation['box_coodi'] = convertcoodi
            annotations.append(annotation)

        box_coords = element.iter(tag="bndbox")

        for box_coord in box_coords:
            xmin = int(box_coord.find("xmin").text)
            ymin = int(box_coord.find("ymin").text)
            xmax = int(box_coord.find("xmax").text)
            ymax = int(box_coord.find("ymax").text)
            # annotation['bndbox'] = [xmin,ymin,xmax,ymax]

            annotation['box_coodi'] = [[xmin, ymin], [xmax, ymin], [xmax, ymax],
                                       [xmin, ymax]]
            annotations.append(annotation)




    bounds = []
    for i in range(len(annotations)):
        box_info_dict = {"points": None, "text": None, "ignore": None}

        box_info_dict["points"] = np.array(annotations[i]['box_coodi'])
        if annotations[i]['name'] == "dnc":
            box_info_dict["text"] = "###"
            box_info_dict["ignore"] = True
        else:
            box_info_dict["text"] = annotations[i]['name']
            box_info_dict["ignore"] = False

        bounds.append(box_info_dict)

    return bounds


def segment_image_by_local_maxima(region_score):

    def valid(box):
        if (box[1][0] - box[0][0]) <= 3 or (box[2][1] - box[0][1]) <= 3 or \
                (box[2][1] - box[0][1]) / (box[1][0] - box[0][0]) > 3 or (box[2][1] - box[0][1]) / (box[1][0] - box[0][0]) < 0.15:
            return False
        return True

    # region_score_color = cv2.applyColorMap(np.uint8(region_score), cv2.COLORMAP_JET)

    region_score = ndi.maximum_filter(region_score, size=3, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    fore = np.uint8(peak_local_max(region_score, min_distance=4,
                                   indices=False, exclude_border=(0, 0)))
    # fore = np.uint8(peak_local_max(region_score, min_distance=int(region_score.shape[0] / (word_num * 3)),
    #                                indices=False, exclude_border=True))

    back = np.uint8(region_score < 0.05)
    unknown = 1 - (fore + back)
    ret, markers = cv2.connectedComponents(fore, connectivity=8)
    markers += 1
    markers[unknown == 1] = 0

    labels = watershed(-region_score, markers, compactness=0.001)
    boxes = []
    width = []
    height = []
    for label in range(2, ret + 1):
        y, x = np.where(labels == label)
        x_max = x.max()
        y_max = y.max()
        x_min = x.min()
        y_min = y.min()
        box = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        box = np.array(box, dtype=np.float32)
        width.append(x_max - x_min)
        height.append(y_max - y_min)

        if not valid(box):
            continue
        boxes.append(box)

    for i in range(2):
        ave = np.mean(np.array(boxes), axis=0)
        w = ave[1][0] - ave[0][0]
        h = ave[2][1] - ave[0][1]
        res = []
        tag = True
        for box in boxes:
            if box[1][0] - box[0][0] < w * 0.3 or box[2][1] - box[0][1] < h * 0.3:
                tag = False
                continue
            res.append(box)
        boxes = res
        if tag:
            return res

    return res


def get_character_boxes(textmap, text_threshold, low_text):
    # prepare data
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8), connectivity=4)

    det = []

    # New
    segmap = np.zeros(textmap.shape, dtype=np.uint8)
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 5: continue

        # thresholding
        # if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map

        segmap[labels==k] = 255

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=1)

        # make box new
        contours, _ = cv2.findContours(segmap[sy:ey, sx:ex], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0].reshape((-1, 2)))
        x += sx
        y += sy
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
        # End

        segmap[sy:ey, sx:ex] = 0

        det.append(box)

    return det


def test_net(
    net,
    image,
    text_threshold,
    link_threshold,
    low_text,
    cuda,
    poly,
    canvas_size=768,
    mag_ratio=1.5,
    generate_character_bboxes=True,
    generate_word_bboxes=True
):
    # resize

    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy().astype(np.float32)
    score_link = y[0, :, :, 1].cpu().data.numpy().astype(np.float32)

    # NOTE
    score_text = score_text[: size_heatmap[0], : size_heatmap[1]]
    score_link = score_link[: size_heatmap[0], : size_heatmap[1]]

    for i in range(3):
        if np.max(score_text) < low_text:
            text_threshold *= 0.5
            link_threshold *= 0.5
            low_text *= 0.5
        else:
            break

    # ----------------------------------------------- Character -------------------------------------------------------#
    if generate_character_bboxes:
        thre = max(low_text - 0.05, 0.05)
        character_boxes = get_character_boxes(score_text, thre, thre)
        # character_boxes = segment_image_by_local_maxima(score_text)
        character_boxes = adjustResultCoordinates4character(character_boxes, ratio_w, ratio_h, image.shape[0], image.shape[1])
    else:
        character_boxes = None
    # ------------------------------------------------- End -----------------------------------------------------------#

    # ------------------------------------------------- Word ----------------------------------------------------------#
    # st = time.time()
    if generate_word_bboxes:
        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly
        )

        # coordinate adjustment
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h, image.shape[0], image.shape[1])
        for k in range(len(polys)):
            polys[k] = np.flip(polys[k], axis=0)
    else:
        boxes, polys = None, None
    # en = time.time()
    # print(en - st)
    # ------------------------------------------------- End -----------------------------------------------------------#
    
    return character_boxes, boxes, polys
