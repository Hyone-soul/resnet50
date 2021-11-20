# author : Soul
# data   : 2021/11/1419:09
from PIL import Image, ImageEnhance, ImageChops
import numpy as np
import random

# 打开原始图片
def open_img(image):
    return Image.open(image, mode='r')


def rotation(image, mode=Image.BICUBIC):
    """
    对图像进行随即任意角度旋转
    ：param mode 邻近插值，双线性插值，双三次B样条插值（default）
    ：param image PIL的图像image
    ：return 旋转处理后的图像
    """
    # random_angle = np.random.randint(1, 360)
    random_angle = 352
    return image.rotate(random_angle, mode)


def shift(image, off_x=0, off_y=0):
    """
    对图像进行x方向与y方向的平移
    ：param off_x x方向偏移量
    ：param off_y y方向偏移量
    ：param image PIL的图像image
    ：return 平移处理后的图像
    """
    return ImageChops.offset(image, off_x, off_y)


def crop(image):
    """
    对图像随意裁剪
    ：param image PIL的图像image
    ：return 剪切之后的图像
    """
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_size = np.random.randint(min(image_width, image_height) * 0.9,
                                      min(image_width, image_height))
    random_region = (
        (image_width - crop_win_size) >> 1,
        (image_height - crop_win_size) >> 1,
        (image_width + crop_win_size) >> 1,
        (image_height + crop_win_size) >> 1
    )
    return image.crop(random_region)


def flip(image, mode=0):
    """
    对图像进行翻转
    ：param image PIL的图像image
    : param mode 翻转模式 mode=0为左右翻转 mode=1为上下翻转
    ：return 剪切之后的图像
    """
    if mode == 0:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)


def contrast_enhancement(image):
    """
    对图像进行对比度增强
    ：param image PIL的图像image
    ：return 对比度增强后的图像
    """
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.3
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def brightness_enhancement(image):
    """
    对图像进行亮度增强
    ：param image PIL的图像image
    ：return 亮度增强后的图像
    """
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.3
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def random_color(image):
    """
    对图像进行颜色抖动
    ：param image PIL的图像image
    ：return 有颜色色差的图像image
    """
    # random_factor = np.random.randint(10, 25) / 10.   # 随机因子
    # color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(9, 14) / 10.
    brightness_image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整亮度
    random_factor = np.random.randint(9, 14) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(10, 17) / 10.
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def gaussian(image, mean=0.2, sigma=0.3):
    """
    对图像进行高斯噪声处理
    ：param image PIL的图像image
    ：return 高斯噪声处理后的的图像image
    """
    def gaussian_noisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪声处理
        ：param im 单通道图像
        ：param mean 偏移量
        ：param sigma 标准差
        ：return
        """
        for i in range(len(im)):
            im[i] += random.gauss(mean, sigma)
        return im

    # 将图像转化为数组
    img = np.array(image)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussian_noisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussian_noisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussian_noisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))


# img_path = r'D:\360download\RP2K_rp2k_dataset\1108_retail_goods_dataset\train\class1790_老恒和蒸鱼豉油\class1790_train4.jpg'
# img = open_img(img_path)
# img.show()
# img_shift = shift(img, 5, 10)
# img_shift.show()
# img_random_color = random_color(img)
# img_random_color.show()
# img_contrast = contrast_enhancement(img)
# img_contrast.show()
# img_brighten = brightness_enhancement(img)
# img_brighten.show()
