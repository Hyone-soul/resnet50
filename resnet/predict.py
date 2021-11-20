# author : Soul
# data   : 2021/11/1217:27

import os
import argparse
import random
import cv2
import numpy as np
import moxing as mox

import mindspore
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model_utils.config import config

from src.resnet import resnet50

random.seed(1)
np.random.seed(1)


def _crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx, :]


def _normalize(img, mean, std):
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def data_preprocess(img_path):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (256, 256))
    img = _crop_center(img, 224, 224)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    img = _normalize(img.astype(np.float32), np.asarray(mean), np.asarray(std))
    img = img.transpose(2, 0, 1)

    return img


def resnet50_predict(args_opt):
    class_num = config.class_num
    local_data_path = '/cache/data'
    ckpt_file_slice = args_opt.checkpoint_path.split('/')
    ckpt_file = ckpt_file_slice[len(ckpt_file_slice)-1]
    local_ckpt_path = '/cache/'+ckpt_file

    # set graph mode and parallel mode
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)

    # data download
    print('Download data.')
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=local_data_path)
    mox.file.copy_parallel(src_url=args_opt.checkpoint_path, dst_url=local_ckpt_path)

    # load checkpoint into net
    net = resnet50(class_num=class_num)
    param_dict = load_checkpoint(local_ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    correct_num = 0
    total_num = 0
    class_total = 0
    class_correct = 0
    for curDir, _, img_files in os.walk(local_data_path):
        if img_files:
            lst = curDir.rsplit('/')
            length = len(lst)
            curImg_class = lst[length - 1].split('_')[1]
            data_cur_serial = int(lst[length - 1].split('_')[0].split('ss')[1])
            for image in img_files:
                total_num += 1
                class_total += 1
                img = data_preprocess(os.path.join(curDir, image))
                res = net(Tensor(img.reshape((1, 3, 224, 224)), mindspore.float32)).asnumpy()
                data_pre_serial = res[0].argmax() + 1
                if data_cur_serial == data_pre_serial:
                    class_correct += 1
                    correct_num += 1
            class_precision = float(class_correct/class_total)
            if class_precision != 1.0:
                print(lst[length - 1])
            class_correct = 0
            class_total = 0
    precison = float(correct_num/total_num)
    print('预测准确率为' + str(precison))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet50 predict.')
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--checkpoint_path', required=True, type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--device_target', type=str, default='Ascend', help='Device target. Default: Ascend.')
    args_opt = parser.parse_args()
    resnet50_predict(args_opt)
    print('ResNet50 prediction success!')