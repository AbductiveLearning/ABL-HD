import PIL.ImageOps
import numpy as np
import cv2
import torch


def read_dict(dictionary_file):
    dictionary = []
    with open(dictionary_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for word in lines:
            word = word.strip()
            dictionary.append(word)
    return dictionary


def split_list(line_box_label_list, pred_word_list):
    line_word_list, cur_idx = [], 0
    for line_box_label in line_box_label_list:
        pred_line_word_list = pred_word_list[cur_idx : cur_idx + len(line_box_label[0])]
        line_word_str = "".join(pred_line_word_list)
        line_word_list.append(line_word_str)
        cur_idx += len(line_box_label[0])
    assert cur_idx == len(pred_word_list)
    return line_word_list


def pred_line_box_label(
    model, line_box_label_list, idx2label, scale_size=96, batch_size=256
):
    if len(line_box_label_list) == 0:
        return []
    img_list = []
    for line_box_label in line_box_label_list:
        for img in line_box_label[0]:
            img_list.append(img)

    # test_data = TorchDataset(img_list=img_list, scale_size=scale_size)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # pred_idx_list = model.predict_classes(test_loader).numpy()

    pred_idx_list = model.predict_imgs_classes(img_list).numpy()
    pred_word_list = [idx2label[idx] for idx in pred_idx_list]
    line_word_pred_list = split_list(line_box_label_list, pred_word_list)
    return line_word_pred_list


def convert_black_white(img, black_thres=0.4):
    colors = img.getcolors()
    black_cnt = 0
    for i in range(15):
        black_cnt += colors[i][0]
    area = img.size[0] * img.size[1]
    # print(black_cnt, area, black_cnt/area)
    if black_cnt / area > black_thres:
        # print("Is black")
        img = PIL.ImageOps.invert(img)
        # img.show()
    return img


def crop_resize(img, points, IMG_SHAPE=(96, 96)):
    """
    points:[x1,y1,x2,y2,...]
    """
    [x1, y1], [x2, y2] = points[0], points[2]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # cropped = img.crop((x1, y1, x2, y2))
    # resized = cropped.resize(IMG_SHAPE)
    # return resized
    cropped = img[y1:y2, x1:x2]
    resized = cv2.resize(cropped, IMG_SHAPE)
    img_pt = torch.from_numpy(resized).unsqueeze(0)
    return img_pt


def get_line_img_list(img, box_pos_char_pos_list):  # ordered_predict
    # img = img.convert("L")
    # img = convert_black_white(img)
    line_img_pos_list = []
    for box_pos_char_pos in box_pos_char_pos_list:
        char_pos_list, line_pos = box_pos_char_pos[:-1], box_pos_char_pos[-1]
        char_img_list = []
        for char_pos in char_pos_list:
            cropped = crop_resize(img, char_pos)  # crop the img according to character
            char_img_list.append(cropped)
        line_img_pos_list.append([char_img_list, line_pos])
    return line_img_pos_list


def predict_imgs_csv(model, img, name, box_pos_char_pos_list, idx2label):
    # time_start_1=time.time()
    line_img_pos_list = get_line_img_list(img, box_pos_char_pos_list)
    # time_end=time.time()
    # print('--- Croping word boxes time cost',time_end-time_start_1,'s')
    # time_start_1=time.time()

    # Predict each word box
    line_word_pred_list = pred_line_box_label(model, line_img_pos_list, idx2label)
    # time_end=time.time()
    # print('--- Predicting each word box time cost',time_end-time_start_1,'s')
    # time_start_1=time.time()

    # Generate csv file
    file_line_list = []
    for line_img_pos, line_word_pred in zip(line_img_pos_list, line_word_pred_list):
        [char_img_list, line_pos] = line_img_pos
        line_pos_str_list = [
            str(pos) for pos in np.array(line_pos).astype(np.int32).flatten()
        ]
        line_pos_str = ",".join(line_pos_str_list)
        line_str = line_pos_str + "," + line_word_pred + "\n"
        file_line_list.append(line_str)

    csv_file = "./outputs/" + name + ".csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("".join(file_line_list))
    # time_end=time.time()
    # print('--- Write csv time cost',time_end-time_start_1,'s')
