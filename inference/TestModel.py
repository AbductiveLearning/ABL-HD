import cv2
import torch
import numpy as np
from craft import CRAFT
from collections import OrderedDict
from inference_boxes import test_net
from match_column_and_character import fast_match
from ordering_utils import predict_order
from word_inference import read_dict, predict_imgs_csv
from word_nn import ResNet34


class Infer(object):
    def __init__(self):
        gpu_idx = torch.cuda.current_device()
        torch.cuda.set_device(gpu_idx)
        map_location = "cuda:%d" % gpu_idx

        # load segmentation model weight
        inference_param = torch.load(
            "weights/KESAR_segmentation_XXX.pth", map_location=map_location
        )

        self.inference_model = CRAFT(amp=True)
        self.inference_model.load_state_dict(
            self.copyStateDict(inference_param["craft"])
        )
        self.inference_model = self.inference_model.cuda().eval()

        # load word model weight
        pretrained_path = "weights/KESAR_word_model_XXX_1.pt"

        self.vocab = read_dict("./dictionary.txt")
        self.idx2label = dict(zip(list(range(len(self.vocab))), self.vocab))
        self.word_model = (
            ResNet34(num_class=len(self.vocab), pretrained_path=pretrained_path)
            .half()
            .cuda()
            .eval()
        )

    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def calc(self, image):
        return np.sum(image < 75) / (image.shape[0] * image.shape[1])

    def eval(self, image_name):
        print(image_name)
        # ------------------- Segmentation Model -------------------
        image = cv2.imread(image_name)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image_gray.shape
        if (
            self.calc(
                image_gray[
                    int(0.25 * height) : int(0.75 * height),
                    int(0.25 * width) : int(0.75 * width),
                ]
            )
            > 0.475
        ):
            image = cv2.bitwise_not(image)
            image_gray = cv2.bitwise_not(image_gray)

        character_boxes, bboxes, polys = test_net(
            self.inference_model,
            image,
            0.55,
            0.3,
            0.5,
            True,
            True,
            768,
            1.5,
            True,
            True,
        )

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for poly in polys:
            cv2.polylines(
                image, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )

        # cv2.imwrite("outputs/" + image_name[:-4] + "_kesar.jpg", image)

        polys, column_level_character_bboxes = fast_match(polys, character_boxes, polys)

        predicted = predict_order(polys, image.shape)

        # new_list is the val in previous pickle file
        box_pos_char_pos_list = []
        for order in predicted:
            box_pos_char_pos_list.append(column_level_character_bboxes[order])

        name = image_name.split("/")[-1].split(".")[0]
        predict_imgs_csv(
            self.word_model, image_gray, name, box_pos_char_pos_list, self.idx2label
        )


if __name__ == "__main__":
    inferencer = Infer()
    # change image path
    inferencer.eval("images/XXX.jpg")
