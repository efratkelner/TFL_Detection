import PIL
from PIL.Image import Image
import numpy as np
from keras import Model

from neural_net import *


def is_tfl(path):
    model = keras.models.load_model(r"C:\Users\efi\Desktop\bootcamp\mobileye\mobileyeProject\model.h5")
    # model = keras.models.load_model('model.h5')
    img = PIL.Image.open(path)
    image_x = 32
    image_y = 32
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = img.reshape(1, 224, 224, 3)
    probability = model.predict(img)[0]
    if max(probability) == 1:
        return True
    return False


def crop_image(x, y, img, c):
    fig, ax = plt.subplots()
    path = r"C:\Users\user\Desktop\bootcamp\mobileye\mobileyeProject\candidates"
    im = PIL.Image.open(img)
    plt.imshow(im)

    left = x - (81 / 2)
    top = y - (81 / 2)
    right = x + (81 / 2)
    bottom = y + (81 / 2)
    try:
        im = im.crop((left, top, right, bottom))
        im.save(path + '\\' + str(c) + ".png")
        plt.cla()
    except:
        print("Fail")
    plt.close(fig)
    return path + '\\' + str(c) + ".png"