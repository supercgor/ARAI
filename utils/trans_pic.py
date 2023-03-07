import numpy as np
import cv2


# def center_crop(img, final_size=(256, 256)):
#     raw_shape = np.array(img.shape)
#     mv = np.subtract(raw_shape, np.array(final_size)) / 2
#     l_s1 = int(mv[0])
#     l_e1 = l_s1 + int(final_size[0])
#     l_s2 = int(mv[1])
#     l_e2 = l_s2 + int(final_size[1])
#     img_crop = img[l_s1:l_e1, l_s2:l_e2]
#     return img_crop
#
#
# def fit_pic(img, final_size=(256, 256)):
#     raw_shape = np.array(np.array(img.shape))
#     height = raw_shape[0]
#     width = raw_shape[1]
#     if height > final_size[0] or width > final_size[1]:
#         dim_diff = abs(width - height)
#         pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
#         if height >= width:
#             pad = (0, 0, pad1, pad2)
#         else:
#             pad = (pad1, pad2, 0, 0)
#         top, bottom, left, right = pad
#         img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
#         img_pad = center_crop(img_pad, final_size=final_size)
#     elif height == final_size[0] and width == final_size[1]:
#         return img
#     else:
#         dif1 = final_size[0] - height
#         dif2 = final_size[1] - width
#         top = dif1 // 2
#         bottom = dif1 - top
#         left = dif2 // 2
#         right = dif2 - left
#         img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
#     return img_pad


def pixel_shift(img, shift):
    dx, dy = shift
    raw_shape = np.array(np.array(img.shape))
    height, width = raw_shape
    target = np.zeros((height, width))
    y, x = np.indices(raw_shape)
    x1 = np.clip(x + dx, 0, width - 1)
    y1 = np.clip(y + dy, 0, height - 1)
    target[y, x] = img[y1, x1]
    return target


# a randomly rectangular place will be cut to zero
# the h, w of the rec box is the normal distribution of the square root(rec_size)
# the initial point of rec is uniform, if the rec touch the boundary, the rec wll be cut.
def cutout(img, rec_size=0.01, max_cut=3):
    height = img.shape[0]
    width = img.shape[1]
    for _ in range(np.random.randint(0, max_cut)):
        change_pixel = img.mean() * abs(np.random.normal(1, 0.2))
        change_pixel = np.clip(np.int8(change_pixel), 0, 255)

        rec_size = np.random.uniform(0, rec_size)
        max_size = int(height * width * rec_size)

        new_height = np.clip(abs(np.random.normal(
            np.sqrt(max_size) * 0.3, np.sqrt(max_size)*0.4)), 1, max_size)
        new_weight = max_size//new_height

        new_height, new_weight = np.random.choice([new_height, new_weight], 2, replace = False)
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        
        m_y = np.clip(new_height + y, 0, height - 1).astype(int)
        y = np.clip(y, 0, height - 1)
        m_x = np.clip(new_weight + x, 0, width - 1).astype(int)
        x = np.clip(x, 0, width - 1)
        img[y:m_y, x:m_x] = change_pixel
    return img


def add_noise(img, max_c=0.1):
    c = abs(np.random.normal(0, max_c))
    c = np.clip(c, 0, max_c)
    noisemode, addmode = np.random.randint(0, [1, 2])
    if noisemode == 0:
        noise = np.random.normal(loc=0, scale=1, size=img.shape)
    elif noisemode == 1:
        noise = 0.5 * c * (img.max() - img.min())
        noise = np.random.uniform(-1 * noise, noise, img.shape)

    if addmode == 0:  # noise overlaid over image
        noisy = np.clip((img + noise * c * 255), 0, 255)
    elif addmode == 1:  # noise multiplied by image
        noisy = np.clip((img * (1 + noise * c)), 0, 255)
    elif addmode == 2:  # noise multiplied by bottom and top half images
        img2 = img / 255 * 2
        noisy = np.clip(np.where(img2 <= 1, (img2 * (1 + noise * c)),
                        (1 - img2 + 1) * (1 + noise * c) * -1 + 2) / 2, 0, 1)
        noisy = noisy * 255
    return noisy

def reduce_bnc(img, b = 0.2, c = 0.1):
    contrast = np.clip(c * 255, 0, 255)
    brightness = np.clip(b * 255, 0, 255)
    output = img * (contrast/127 + 1) - contrast + brightness
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

if __name__ == '__main__':
    img = np.random.random((128, 128))
