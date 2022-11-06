from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
import os
from custom_utils_v2 import *
from detectron2.utils.visualizer import ColorMode
import glob
from detectron2.structures.instances import Instances
import cv2
import csv
from detectron2.utils.visualizer import Visualizer
import argparse
from utils_predictions_extraction import *


dir_name = os.getcwd()
test_dir = dir_name + '\\DATASET\\images\\test'


DEFAULT_CFG = dir_name + "\\output\\config.yaml"
DEFAULT_MODEL = dir_name + "\\output\\model_final.pth"
SAVE_DIR = dir_name + "\\detections"



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=DEFAULT_MODEL, help='model_final.pth path', required=True)
    parser.add_argument('--cfg', type=str, default=DEFAULT_CFG, help='config.yaml path', required=True)
    parser.add_argument('--data_test', type=str, default=test_dir, help='test images path')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='score thresh test')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='save directory path')
    parser.add_argument('-l','--list', nargs='+', help='Liste maximal zu erreichenden Punkten', required=True)
    return parser.parse_args()


def main(opt):
    
    DEFAULT_CFG = opt.cfg
    DEFAULT_MODEL = opt.weights
    SAVE_DIR = opt.save_dir+"\\"
    test_dir = opt.data_test
    
    print('savedir ', SAVE_DIR)
    cfg = get_cfg()
    cfg.merge_from_file(DEFAULT_CFG)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.OUTPUT_DIR = SAVE_DIR
    cfg.MODEL.WEIGHTS = DEFAULT_MODEL
    predictor = DefaultPredictor(cfg)
    
    final_results = []
    results_csv = []

    exists_directory(SAVE_DIR)
    list_points = opt.list # [7,5,2,13,13,8,9,6,27,90]
    list_points = list_points[0].split(",")
    list_points = list(map(int, list_points))
   
    for i in get_list_images(test_dir):
        filename = i.split('\\')[-1]
        im = cv2.imread(i)
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        name = filename.split('/')
        name = name[len(name)-1]
        unboxed_results = csv_result(name, outputs)
        
        for i in  range(len(unboxed_results)):
            if len(unboxed_results[i][0]) == 2:
                if is_same_digit(unboxed_results, i, 0, 1):
                    unboxed_results = get_digit_with_respect_to_score(unboxed_results, i, 0, 1, list_points[i])
            if len(unboxed_results[i][0]) == 2:
                if exists_element_above(unboxed_results, i, 0, 1):
                    unboxed_results = clean_y_duplicates(unboxed_results, i, 0, 1)
            
            if len(unboxed_results[i][0]) == 3:
                if is_same_digit(unboxed_results, i, 0, 1):
                    unboxed_results = get_digit_with_respect_to_score(unboxed_results, i, 0, 1, list_points[i])
            if len(unboxed_results[i][0]) == 3:
                if is_same_digit(unboxed_results, i, 1, 2):
                    unboxed_results = get_digit_with_respect_to_score(unboxed_results, i, 1, 2, list_points[i])

            if len(unboxed_results[i][0]) == 3:  
                if exists_element_above(unboxed_results, i, 0, 1):
                    unboxed_results = clean_y_duplicates(unboxed_results, i, 0, 1)
            if len(unboxed_results[i][0]) == 3:
                if exists_element_above(unboxed_results, i, 1, 2):
                    unboxed_results = clean_y_duplicates(unboxed_results, i, 1, 2)

            if len(unboxed_results[i][0]) == 4:
                if is_same_digit(unboxed_results, i, 0, 1):
                    unboxed_results = get_digit_with_respect_to_score(unboxed_results, i, 0, 1,list_points[i])
            
            if len(unboxed_results[i][0]) == 4:
                if is_same_digit(unboxed_results, i, 1, 2):
                    unboxed_results = get_digit_with_respect_to_score(unboxed_results, i, 1, 2, list_points[i])
            if len(unboxed_results[i][0]) == 4:
                if is_same_digit(unboxed_results, i, 2, 3):
                    unboxed_results = get_digit_with_respect_to_score(unboxed_results, i, 2, 3, list_points[i])

            if len(unboxed_results[i][0]) == 4:  
                if exists_element_above(unboxed_results, i, 0, 1):
                    unboxed_results = clean_y_duplicates(unboxed_results, i, 0, 1)
            if len(unboxed_results[i][0]) == 4:
                if exists_element_above(unboxed_results, i, 1, 2):
                    unboxed_results = clean_y_duplicates(unboxed_results, i, 1, 2)
            if len(unboxed_results[i][0]) == 4:
                if exists_element_above(unboxed_results, i, 2, 3):
                    unboxed_results = clean_y_duplicates(unboxed_results, i, 2, 3)
                
            extracted = extract_points_per_task(unboxed_results, list_points, name)
        results_csv = []
        results_csv.append(name)
        for e in extracted:
            results_csv.append(e)
        results_csv.append(is_sum_correct(extracted))

        with open(SAVE_DIR + "detection_results.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow(results_csv)
        cv2.imwrite(SAVE_DIR + filename, out.get_image()[:, :, ::-1])

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

