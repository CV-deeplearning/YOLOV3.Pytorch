from __future__ import division
import os
import sys
import cv2
import glob
from tqdm import tqdm
from PIL import Image
import torch
import time
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
from models import *


class YoloDetect(object):
    def __init__(self, model_def, weights_path, conf_thres=0.7, nms_thres=0.4, img_size=416):
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.img_size = img_size
        self.model = Darknet(model_def, img_size).to(self.device)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def process(self, img_path):
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)
        original_shape = img.shape[1:]
        img, _ = self.pad_to_square(img, 0)
        img = self.resize(img, self.img_size)
        input_img = Variable(img.type(self.Tensor))
        input_img = input_img.unsqueeze(0)
        return input_img, original_shape

    def detect(self, img_path):
        input_img, original_shape = self.process(img_path)
        with torch.no_grad():
            start = time.time()
            detections = self.model(input_img)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            detections = self.rescale_boxes(detections[0], self.img_size, original_shape)
            end = time.time()
            print("use time:")
            print(end-start)
        return detections

    def pad_to_square(self, img, pad_value):
        c, h, w = img.shape
        dim_diff = np.abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
        # Add padding
        img = F.pad(img, pad, "constant", value=pad_value)
        return img, pad

    def resize(self, image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image

    def rescale_boxes(self, boxes, current_dim, original_shape):
        """ Rescales bounding boxes to the original shape """
        orig_h, orig_w = original_shape
        # The amount of padding that was added
        pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
        pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
        # Image height and width after padding is removed
        unpad_h = current_dim - pad_y
        unpad_w = current_dim - pad_x
        # Rescale bounding boxes to dimension of original image
        boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
        boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
        boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
        return boxes


def draw(img_path, out_dir, detections):
    class_list = ["people", "box", "face"]
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    save_path = os.path.join(out_dir, img_name)
    if detections is None:
        cv2.imwrite(save_path, img)
    for x, y, x2, y2, conf, cls_conf, cls_pred in detections:
        p1 = (int(x), int(y))
        p2 = (int(x2), int(y2))
        width = p2[0] - p1[0]
        height = p2[1] - p1[1]
        p3 = (max(p1[0], 15), max(p1[1], 15))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 2)
        title = "%s:%.2f" % (class_list[int(cls_pred)], conf)
        cv2.putText(img, title, p3, cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
    cv2.imwrite(save_path, img)



if __name__ == "__main__":
    model_def = "config/yolov3-custom.cfg"
    weights_path = "checkpoints/yolov3_ckpt_95.pth"
    out_dir = "output"
    
    img_list = glob.glob("data/samples/*.jpg")
    detector = YoloDetect(model_def, weights_path)
    for img_path in tqdm(img_list):
        detections = detector.detect(img_path)
        draw(img_path, out_dir, detections)
    
