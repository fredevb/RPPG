import numpy as np
import math
import cv2

def crop_image(image, xmin, ymin, width, height):
    return image[ymin:ymin+height, xmin:xmin+width]

def sub_bounding_box(width, height, percent_xmin, percent_ymin, percent_width, percent_height, zoom):
    zoom_x, zoom_y = zoom
    w, h = int(percent_width * width), int(percent_height * height)
    xm, ym, = int(percent_xmin * width), int(percent_ymin * height)
    new_w, new_h = int(w/zoom_x), int(h/zoom_y)
    new_xm, new_ym = int(xm+(w-new_w)/2), int(ym+(h-new_h)/2)
    return (new_xm, new_ym, new_w, new_h)

def iter_blocks(w, h, n_x, n_y):
    block_w = math.floor(w/n_x)
    block_h = math.floor(h/n_y)
    for i_y in range(0, n_y):
        for i_x in range(0, n_x):
            x_min = i_x*block_w
            y_min = i_y*block_w
            yield x_min, y_min, block_h, block_w

def get_image_blocks(image, n_x, n_y):
    return [
        crop_image(image, x_min, y_min, h, w)
        for image, x_min, y_min, h, w
        in iter_blocks(image.shape[0], image.shape[1], n_x, n_y)
        ]

def get_bounding_box(polygon):
    min_x, max_x, min_y, max_y = -1, -1, -1, -1
    for point in polygon:
        x, y = point[0], point[1]
        if x < min_x or min_x == -1:
            min_x = x
        if max_x < x or max_x == -1:
            max_x = x
        if y < min_y or min_y == -1:
            min_y = y
        if max_y < y or max_y == -1:
            max_y = y
    h = max_y - min_y
    w = max_x - min_x
    return min_x, min_y, w, h

def get_bounding_box_points(xmin, ymin, width, height):
    ret = []
    for y in [ymin, ymin + height]:
        for x in [xmin, xmin + width]:
            ret.append([x, y])
    # Swap
    temp = ret[2]
    ret[2] = ret[3]
    ret[3] = temp
    return ret

def cut_polygon(image, polygon):
    xmin, ymin, w, h = get_bounding_box(polygon)
    if w == 0 or h == 0:
        return np.full((1,1,3), np.nan, dtype=np.uint8)
    mask = np.full((h,w,3), np.nan, dtype=np.uint8)
    mask_polygon = np.array([[point[0] - xmin, point[1] - ymin] for point in polygon])
    cv2.fillPoly(mask, [mask_polygon], color=(255, 255, 255))
    crop = crop_image(image, xmin, ymin, w, h)
    mask = cv2.bitwise_and(mask, crop)
    return mask

def percentage_coords_to_image_coords(percentage_coords_array, w, h):
    basis = np.array([[w, 0], [0, h]])
    points = percentage_coords_array @ basis # Rules of transposition used.
    return [np.round(p).astype(int) for p in points]