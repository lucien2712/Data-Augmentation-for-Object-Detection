
import os
import json
import pandas as pd
import numpy as np
from PIL import Image

def create_dataset_df(json_path, max_images=10000):

    with open(json_path, 'r') as f:
        labels_data = json.load(f)

    total_images = min(len(labels_data), max_images)
    dataset = []

    for item in labels_data[:total_images]:
        dataset.append({
            'image_filename': item['image'],
            'labels': item['labels'],
            'height': item['height'],
            'width': item['width'],
            'bboxes': item['bboxes']
        })

    return pd.DataFrame(dataset)

def load_and_preprocess_image(image_path):

    img = Image.open(image_path).convert('RGB')
    return img

def normalize_boxes(boxes, original_width, original_height):
 
    normalized_boxes = []
    for box in boxes:
        box = [float(coord) for coord in box]
        x1, y1, x2, y2 = box
        normalized_boxes.append([
            x1/original_width,
            y1/original_height,
            x2/original_width,
            y2/original_height
        ])
    return normalized_boxes

def create_output_dir(output_dir):
  
    os.makedirs(output_dir, exist_ok=True) 