#!/usr/bin/env python3
"""Module 0-yolo.py
Yolo v3 algorithm to perform object detection
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """Loads the classes from a file"""
        with open(classes_path, 'r') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def process_outputs(self, outputs, image_size):
        """Returns a tuple of (boxes, box_confidences, box_class_probs)"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        for index, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            box = output[..., :4]
            for i in range(grid_height):
                for j in range(grid_width):
                    for k in range(anchor_boxes):
                        pw, ph = self.anchors[index][k]
                        t_x, t_y, t_w, t_h = box[i, j, k]

                        # Calculate box coordinates
                        bx = (1 / (1 + np.exp(-t_x)) + j) / grid_width
                        by = (1 / (1 + np.exp(-t_y)) + i) / grid_height
                        bw = (1 / (np.exp(-t_w)) * pw) / image_size[1]
                        bh = (1 / (np.exp(-t_h)) * ph) / image_size[0]

                        x1 = bx - bw / 2
                        y1 = by - bh / 2
                        x2 = x1 + bw
                        y2 = y1 + bh

                        box[i, j, k] = [x1, y1, x2, y2]

            boxes.append(box)
            box_confidences.append(output[..., 4:5])
            box_class_probs.append(output[..., 5:])

        return boxes, box_confidences, box_class_probs
