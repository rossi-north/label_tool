from pathlib import Path
import argparse

import cv2
import numpy as np
from loguru import logger
from ultralytics import YOLO

from annotation import AnnotationBase
from img_process import get_card_cnt

parser = argparse.ArgumentParser()
parser.add_argument('-img', '--image_folder', type=str, required=True, help='Path to the image folder')
parser.add_argument('-w', '--weight_path', type=str, required=True, help='Path to the ultralytics model path')
parser.add_argument('-v', '--version', type=str, required=True, help='X-anylabeling version')
args = parser.parse_args()

img_folder = Path(args.image_folder)
weight_path = Path(args.weight_path)
if not img_folder.exists():
    assert False, "image folder not exist."
if not weight_path.exists():
    assert False, "iweight not exist."

model = YOLO(weight_path, task='segment')

WHITE_LOWER = np.array([0, 0, 221])
WHITE_UPPER = np.array([180, 30, 255])

for img_p in img_folder.glob('*.jpg'):
    logger.info(f'read {img_p}')
    img = cv2.imread(str(img_p))
    img_name = img_p.name
    h, w, _ = img.shape
    img_anno = AnnotationBase(args.version, img_name, h, w)

    # inference image
    results = model.predict(img, retina_masks=True, agnostic_nms=True, verbose=False)[0]
    names = results.names

    #print(len(img_anno.shapes))

    for i, bbox in enumerate(results.boxes):
        label_name = names[int(bbox.cls.item())]
        x1, y1, x2, y2 =bbox.xyxy.cpu().numpy().astype(np.int32)[0]
        new_x1 = max(0, x1-5)
        new_y1 = max(0, y1-5)
        new_x2 = min(w-1, x2+5)
        new_y2 = min(h-1, y2+5)
        card = img[new_y1:new_y2, new_x1:new_x2, :]

        card_cnt = get_card_cnt(card, WHITE_LOWER, WHITE_UPPER)
        card_cnt = card_cnt + np.array([new_x1, new_y1], dtype=np.int32)
        card_cnt = card_cnt.reshape(-1, 2).astype(np.float32).tolist()

        if card_cnt is None:
            continue

        img_anno.add_shape(label_name, card_cnt)
        

    if len(img_anno.shapes) == 0:
        continue

    img_anno.save_to_json(img_p.with_suffix('.json'))
    break