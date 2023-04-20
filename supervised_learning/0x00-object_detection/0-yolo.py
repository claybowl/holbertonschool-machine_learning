#!/usr/bin/env python3
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
