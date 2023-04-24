#!/usr/bin/env python3
"""module 7-yolo.py
"""
import os
import numpy as np
import cv2
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

    @staticmethod
    def load_images(folder_path):
        """Loads images from a folder"""
        image_paths = [os.path.join(folder_path, img) for img in os.listdir(
            folder_path) if img.endswith(('.jpg', '.jpeg', '.png'))]
        images = [cv2.imread(image_path) for image_path in image_paths]
        return images, image_paths

    def process_outputs(self, outputs, image_size):
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
                        bw /= self.model.input.shape[1]
                        bh /= self.model.input.shape[2]
                        x1 = (bx - (bw / 2)) * image_width
                        y1 = (by - (bh / 2)) * image_height
                        x2 = (bx + (bw / 2)) * image_width
                        y2 = (by + (bh / 2)) * image_height
                        boxes[i][cy, cx, b] = [x1, y1, x2, y2]
        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
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
        x1 = np.maximum(box1[0], boxes[0])
        y1 = np.maximum(box1[1], boxes[1])
        x2 = np.minimum(box1[2], boxes[2])
        y2 = np.minimum(box1[3], boxes[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes_area = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])

        union_area = box1_area + boxes_area - intersection_area

        return intersection_area / union_area

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies non-maximum suppression to the filtered boxes.

        If use_tf is True, it uses TensorFlow's non_max_suppression implementation.
        Otherwise, it uses the provided custom implementation.
        """
        unique_classes = np.unique(box_classes)
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)
            cls_boxes = filtered_boxes[idxs]
            cls_box_scores = box_scores[idxs]

            while len(cls_boxes) > 0:
                max_score_idx = np.argmax(cls_box_scores)
                box_predictions.append(cls_boxes[max_score_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_box_scores[max_score_idx])

                iou_scores = [self.intersection_over_union(cls_boxes[max_score_idx],
                                                           box) for box in cls_boxes]
                to_remove = np.where(np.array(iou_scores) > self.nms_t)
                cls_boxes = np.delete(cls_boxes, to_remove, axis=0)
                cls_box_scores = np.delete(cls_box_scores, to_remove, axis=0)

        return np.array(box_predictions), np.array(predicted_box_classes), np.array(predicted_box_scores)

    def preprocess_images(self, images):
        """Preprocess images for the YOLO model"""
        input_h = self.model.input_shape[1]
        input_w = self.model.input_shape[2]
        ni = len(images)
        pimages = np.empty((ni, input_h, input_w, 3))
        image_shapes = np.empty((ni, 2))

        for i, img in enumerate(images):
            image_shapes[i] = img.shape[:2]
            resized_img = cv2.resize(
                img, (input_w, input_h), interpolation=cv2.INTER_CUBIC)
            pimages[i] = resized_img / 255

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Displays the image with all boundary boxes, class names, and box scores"""
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "{} {:.2f}".format(
                self.class_names[box_classes[i]], box_scores[i])
            cv2.putText(image, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            save_path = os.path.join('detections', os.path.basename(file_name))
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """Predicts objects in images within the folder and displays them using the show_boxes method"""
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        predictions = []
        for i, pimage in enumerate(pimages):
            outputs = self.model.predict(pimage[np.newaxis, ...])
            boxes, box_confidences, box_class_probs = self.process_outputs(
                outputs, image_shapes[i])
            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes, box_confidences, box_class_probs)
            box_predictions, predicted_box_classes, predicted_box_scores = self.non_max_suppression(
                filtered_boxes, box_classes, box_scores)
            predictions.append(
                (box_predictions, predicted_box_classes, predicted_box_scores))
            self.show_boxes(
                images[i], box_predictions, predicted_box_classes, predicted_box_scores, image_paths[i])

        return predictions, image_paths
