import cv2
import numpy as np

def gamma_correction(img: np.ndarray, gamma: float, channel: int, gray: bool=False) -> np.ndarray:
    look_up_table = (pow(np.arange(256) / 255, gamma) *
                     255).astype(np.uint8).reshape((1, 256))

    if not gray:
        # to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, channel] = cv2.LUT(hsv[:, :, channel], look_up_table)
        res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        res = cv2.LUT(img, look_up_table)

    return res

def get_card_cnt(img: np.ndarray, white_lower: np.ndarray, white_upper: np.ndarray, smoothing: float=0.005):
    
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=33)
    img = cv2.divide(img, gaussian, scale=255)
    gamma_card = gamma_correction(img, 0.25, channel=1)
    hsv_img = cv2.cvtColor(gamma_card, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, white_lower, white_upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None
    
    # get max contour
    max_card_cnt = sorted(cnts, key=lambda i: cv2.contourArea(i), reverse=True)[0]
    epsilon = smoothing * cv2.arcLength(max_card_cnt, True)
    approx = cv2.approxPolyDP(max_card_cnt, epsilon, True)

    return approx