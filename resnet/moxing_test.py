# author : Soul
# data   : 2021/11/1117:44

import os
import moxing as mox
from src.dataset import create_dataset2 as create_dataset
import argparse


def train(args_opt):
    # print(os.listdir('obs://soul-resnet-new/datatset/train'))
    dst_data_path = '/cache/data'
    test_data_path = '/cache/test'
    mox.file.copy_parallel(src_url=args_opt.data_url, dst_url=dst_data_path)
    mox.file.copy_parallel(src_url=args_opt.test_url, dst_url=test_data_path)
    # train_dataset = create_dataset(dataset_path=dst_data_path, do_train=True, repeat_num=1, batch_size=32)
    # step_size = train_dataset.get_dataset_size()
    print(os.listdir(dst_data_path))
    print('--------------------------------------')
    print(os.listdir(test_data_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--data_url', type=str, default=None, help='the data url')
    parser.add_argument('--test_url', type=str, default=None, help='the data url')
    args_opt = parser.parse_args()
    train(args_opt)
