# author : Soul
# data   : 2021/11/1513:01
from DataEnhance import open_img, rotation, shift, contrast_enhancement, random_color, gaussian
import os
import math


data_dir = 'D:\\360download\\RP2K_rp2k_dataset\\1108_retail_goods_dataset\\train'
f = open("image_sample.txt", encoding='utf-8')  # 打开记录的txt文本，里面有需要数据增强的类名称
class_name = f.readline().replace('\n', '')  # 调用文件的 readline()方法，去除换行符
while class_name:
    print(class_name)
    classImage_path = os.path.join(data_dir, class_name)   # 生成类文件夹访问路径
    image_list = os.listdir(classImage_path)  # 类文件夹中所对应的图片列表
    list_len = len(image_list)                # 计算列表长度，对列表中的1/3数量图片做数据增强
    use_len = math.ceil(list_len/4)
    count_num = 0
    for image_name in image_list:
        count_num += 1
        image_path = os.path.join(classImage_path, image_name)
        src_img = open_img(image_path)

        img_rotate = rotation(src_img)  # 1、旋转
        img_rotate.save(os.path.join(classImage_path, 'rotateSample' + str(count_num) + '.jpg'))  # 生成旋转后的图片，并保存到相应类文件夹

        img_contrast = contrast_enhancement(src_img)  # 2、对比度提高
        img_contrast.save(os.path.join(classImage_path, 'contrastSample' + str(count_num) + '.jpg'))

        img_random_color = random_color(src_img)  # 3、随机抖动
        img_random_color.save(os.path.join(classImage_path, 'randomColorSample' + str(count_num) + '.jpg'))

        img_gaussian = gaussian(src_img)  # 4、高斯模糊
        img_gaussian.save(os.path.join(classImage_path, 'gaussianSample' + str(count_num) + '.jpg'))

        img_shift = shift(src_img, 5, 10)  # 5、平移变换
        img_shift.save(os.path.join(classImage_path, 'shiftSample' + str(count_num) + '.jpg'))
        if count_num == use_len:
            break
    class_name = f.readline().replace('\n', '')

f.close()
