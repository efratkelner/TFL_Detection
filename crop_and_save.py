from random import random

try:
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image

except ImportError as e:
    print(f"Installation error: {e}")
    raise

from PIL import Image

plt.rcParams.update({'figure.max_open_warning': 0})

tfl_counter = 0
not_tfl_counter = 0


def crop_image(x, y, img, tav):
    fig, ax = plt.subplots()
    global tfl_counter
    global not_tfl_counter
    tfl_path = r"C:\Users\efi\Desktop\Bootcamp\mobileye\val\tfl"
    not_tfl_path = r'C:\Users\efi\Desktop\Bootcamp\mobileye\val\not tfl'
    im = Image.open(img)
    plt.imshow(im)

    left = x - (81 / 2)
    top = y - (81 / 2)
    right = x + (81 / 2)
    bottom = y + (81 / 2)

    try:
        im = im.crop((left, top, right, bottom))
        if tav == '1':
            im.save(tfl_path + '\\' + tav + str(tfl_counter) + ".png")
            tfl_counter += 1
        else:
            im.save(not_tfl_path + '\\' + tav + str(not_tfl_counter) + ".png")
            not_tfl_counter += 1
        plt.cla()
    except:
        print("Fail")
    plt.close(fig)


def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (_x, _y)


def crop_and_save(image_path, image, objs, fig_num=None):
    path = image
    plt.figure(fig_num).clf()
    in_tfl = []
    out_tfl = []
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            polygon = Polygon(poly)
            center = centroid(poly)
            in_tfl += [center]

        while len(out_tfl) < len(in_tfl):
            x = random.randint(0, image.shape[0])
            y = random.randint(0, image.shape[1])
            point = Point(x, y)

            if not polygon.contains(point):
                out_tfl += [(x, y)]
    for p in in_tfl:
        crop_image(p[0], p[1], image_path, '1')
    for p in out_tfl:
        crop_image(p[0], p[1], image_path, '0')


def find_tfl_lights(image_path, json_path=None, fig_num=None):
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    crop_and_save(image_path, image, objects, fig_num)


def main():
    path_png = r"C:\Users\efi\Desktop\Bootcamp\mobileye\leftImg8bit\val"
    path_json = r"C:\Users\efi\Desktop\Bootcamp\mobileye\gtFine_trainvaltest\gtFine\val"
    for folder in os.listdir(path_png):
        path_ = path_png + "\\" + folder
        json_fn = path_json + "\\" + folder
        for img_name in os.listdir(path_):
            if img_name.endswith(".png"):
                img_path = path_ + '\\' + img_name
                json_p = json_fn + '\\' + img_name
                if img_path.endswith('_leftImg8bit.png'):
                    json_p = json_p.replace('_leftImg8bit.png', '_gtFine_polygons.json')
                    find_tfl_lights(img_path, json_p)


if __name__ == '__main__':
    main()
