from copyreg import pickle
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
setup_logger()
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from custom_utils_v2 import *
import os
import torch
import argparse


config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

save_dir = "./output/detection_model"
num_classes = 11

device = "cuda" # "cpu"
train_dataset_name = "dataset_train"
dirname = os.getcwd()
train_dir = dirname + '\\DATASET\\images\\train' 
val_dir = dirname + '\\DATASET\\images\\valid'
path_annotations = dirname + '\\DATASET\\annotations'


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_test', type=str, default=test_dir, help='test images path')
    parser.add_argument('--data_train', type=str, default=train_dir, help='train images path')
    parser.add_argument('--data_valid', type=str, default=val_dir, help='validation images path')
    parser.add_argument('--save_dir', type=str, default=save_dir, help='output directory path')
    parser.add_argument('--annotations', type=str, default=path_annotations, help='annotations path')
    parser.add_argument('--num_cls', type=int, default=11, help='number of classes int')
    parser.add_argument('--workers', type=int, default=4, help='numver workers int')
    parser.add_argument('--batch', type=int, default=6, help='batch size int')
    parser.add_argument('--epoch', type=int, default=1100, help='number of epochs int')
    parser.add_argument('--lr', type=int, default=0.01, help='learning rate int')
    parser.add_argument('--config_url',
                        type=str, default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml path')
    parser.add_argument('--checkpoint_url',
                        type=str, default='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml path')

    return parser.parse_args()


def main(opt):

    config_file = opt.config_url
    checkpoint_url = opt.checkpoint_url
    train_dir = opt.data_train
    val_dir = opt.data_valid
    num_classes = opt.num_cls
    save_dir = opt.save_dir
    num_workers = opt.workers
    batch_size = opt.batch
    path_annotations = opt.annotations
    lr = opt.lr

    exists_directory(save_dir)

    epochs = opt.epoch
    cfg = get_train_configs(config_file, checkpoint_url, train_dir, val_dir, path_annotations, num_classes, device, save_dir, num_workers, batch_size, epochs, lr)

    trainer = CocoTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Download Config File
    f= open(save_dir+"/config.yaml","w")
    f.write(cfg.dump())
    f.close()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)