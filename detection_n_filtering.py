
try:
    import matplotlib.image as mpimg
    from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters
    import cv2
    from scipy import signal as sg
    from PIL import Image

except ImportError as e:
    print(f"Installation error: {e}")
    raise


import numpy as np
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

img_1= '6.png'
img_path = img_1


def print_image(img_path):
  img = load_img(img_path, target_size=(150, 150))  # a PIL image
  x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
  imgplot = plt.imshow(array_to_img(x))

  x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)
  # The .flow() command below generates batches of randomly transformed images
  # It will loop indefinitely, so we need to `break` the loop at some point!


print_image(img_path)

y,x = np.ogrid[-2:2+1, -2:2+1]
mask = x**2+y**2 <= 3**0
mask = 1*mask.astype(float)
mask[mask == 1] = 8/9
mask[mask == 0] = -1/8

print(mask)


def detect_peaks(img, threshold, neighborhood_size):
    """
    Detects local maxima (peaks) in an image using a neighborhood filter.

    :param img: The input grayscale image as a 2D NumPy array.
    :param threshold: Minimum intensity difference required for a peak to be considered significant.
    :param neighborhood_size: The size of the neighborhood used for peak detection.
    :return: tuple[list[float], list[float]]:
            - x (list): List of x-coordinates of detected peaks.
            - y (list): List of y-coordinates of detected peaks.
    """
    data_max = filters.maximum_filter(img, neighborhood_size)
    maxima = (img == data_max)
    data_min = filters.minimum_filter(img, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)

    plt.autoscale(False)
    return x, y


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    img = cv2.imread(img_path)

    info = np.iinfo(img.dtype)  # Get the information of the incoming image type
    # one channel 3D
    red_image = img.copy()

    red_image[:, :, 0] = 0
    red_image[:, :, 1] = 0

    # one channel 2D
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    plt.imshow(Image.fromarray(r, 'L'))

    # create mask
    kernel = mask

    g = g.astype(np.float64) / info.max
    r = r.astype(np.float64) / info.max

    img_b = sg.convolve(b, kernel, mode='same', method='auto')
    img_g = sg.convolve(g, kernel, mode='same', method='auto')
    img_r = sg.convolve(r, kernel, mode='same', method='auto')

    plt.imshow(Image.fromarray(img_g, 'L'))

    return img_r, img_g


def main():
    img_r, img_g = find_tfl_lights(img_1)
    red_x, red_y = detect_peaks(img_r, 1.5, 4)
    green_x, green_y = detect_peaks(img_g, 1.6, 4)
    img = mpimg.imread(img_1)
    plt.imshow(img)
    plt.plot(red_x, red_y, 'rx', markersize=4)
    plt.plot(green_x, green_y, 'g+', markersize=4)


if __name__ == '__main__':
    main()
