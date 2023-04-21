#!/usr/bin/env python3
"""module 3-yolo.py
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
        """Process Darknet outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i in range(len(outputs)):
            boxes.append(outputs[i][..., :4])
            box_confidences.append(1 / (1 + np.exp(-outputs[i][..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-outputs[i][..., 5:])))
        image_height, image_width = image_size
        for i in range(len(boxes)):
            grid_width = outputs[i].shape[1]
            grid_height = outputs[i].shape[0]
            anchor_boxes = outputs[i].shape[2]
            for cy in range(grid_height):
                for cx in range(grid_width):
                    for b in range(anchor_boxes):
                        tx, ty, tw, th = boxes[i][cy, cx, b]
                        pw, ph = self.anchors[i][b]
                        bx = (1 / (1 + np.exp(-tx))) + cx
                        by = (1 / (1 + np.exp(-ty))) + cy
                        bw = pw * np.exp(tw)
                        bh = ph * np.exp(th)
                        bx /= grid_width
                        by /= grid_height
                        bw /= self.model.input.shape[1].value
                        bh /= self.model.input.shape[2].value
                        x1 = (bx - (bw / 2)) * image_width
                        y1 = (by - (bh / 2)) * image_height
                        x2 = (bx + (bw / 2)) * image_width
                        y2 = (by + (bh / 2)) * image_height
                        boxes[i][cy, cx, b] = [x1, y1, x2, y2]
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the boxes based on class scores,
        class probabilities, and a predefined threshold.
        """
        filtered_boxes, box_classes_list, box_scores_list = None, [], []
        for i in range(len(boxes)):
            new_box_score = box_confidences[i] * box_class_probs[i]
            new_box_class = np.argmax(new_box_score, axis=-1)
            new_box_score = np.max(new_box_score, axis=-1)

            box_classes_list.append(new_box_class.reshape(-1))
            box_scores_list.append(new_box_score.reshape(-1))

        box_scores_all = np.concatenate(box_scores_list)
        box_classes_all = np.concatenate(box_classes_list)
        box_mask = box_scores_all >= self.class_t

        filtered_boxes = np.concatenate(
            [box.reshape(-1, 4) for box in boxes], axis=0)
        filtered_boxes = filtered_boxes[box_mask]

        box_classes = box_classes_all[box_mask]
        box_scores = box_scores_all[box_mask]

        return filtered_boxes, box_classes, box_scores

    def intersection_over_union(self, box1, boxes):
        """Calculate the Intersection over Union (IoU) for a given box and multiple other boxes."""
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])

        intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores, use_tf=True):
        """
        Applies non-maximum suppression to the filtered boxes.

        If use_tf is True, it uses TensorFlow's non_max_suppression implementation.
        Otherwise, it uses the provided custom implementation.
        """
        if use_tf:
            indices = tf.image.non_max_suppression(filtered_boxes,
                                                   box_scores,
                                                   max_output_size=4096,
                                                   iou_threshold=self.nms_t,
                                                   score_threshold=self.class_t)
            indices = indices.numpy()

            box_predictions = filtered_boxes[indices]
            predicted_box_classes = box_classes[indices]
            predicted_box_scores = box_scores[indices]
        else:
            box_predictions = []
            predicted_box_classes = []
            predicted_box_scores = []

            for c in set(box_classes):
                idxs = np.where(box_classes == c)
                class_boxes = filtered_boxes[idxs]
                class_box_scores = box_scores[idxs]

                while len(class_boxes) > 0:
                    max_idx = np.argmax(class_box_scores)
                    box_predictions.append(class_boxes[max_idx])
                    predicted_box_classes.append(c)
                    predicted_box_scores.append(class_box_scores[max_idx])

                    class_boxes = np.delete(class_boxes, max_idx, axis=0)
                    class_box_scores = np.delete(
                        class_box_scores, max_idx, axis=0)

                    if len(class_boxes) == 0:
                        break

                    iou = self.intersection_over_union(
                        box_predictions[-1], class_boxes)
                    iou_mask = iou < self.nms_t

                    class_boxes = class_boxes[iou_mask]
                    class_box_scores = class_box_scores[iou_mask]

            box_predictions = np.array(box_predictions)
            predicted_box_classes = np.array(predicted_box_classes)
            predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores
