import os
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from aicrowd_wrapper import AIcrowdWrapper


def check_data(datafolder):
    """
    Checks if the data is downloaded and placed correctly
    """
    imagefolder = os.path.join(datafolder, 'inputs')
    annotationsfolder = os.path.join(datafolder, 'semantic_annotations')
    dl_text = ("Please download the public data from"
               "\n https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/semantic-segmentation/dataset_files"
               "\n And unzip it with ==> unzip <zip_name> -d public_dataset")
    if not os.path.exists(imagefolder):
        raise NameError(f'No folder named {imagefolder} \n {dl_text}')
    if not os.path.exists(annotationsfolder):
        raise NameError(f'No folder named {annotationsfolder} \n {dl_text}')

def caclulate_dice_nd(class_annotation, class_prediction):
    numer = 2.0 * np.sum(class_annotation & class_prediction)
    denom = np.sum(class_annotation) + np.sum(class_prediction)
    return numer, denom

def calculate_iou_nd(class_annotation, class_prediction):
    numer = np.sum(class_annotation & class_prediction)
    denom = np.sum(class_annotation | class_prediction)
    return numer, denom

class SegmentationMetrics:
    def __init__(self):
        self.class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        nd_dict = {"numer": 0, "denom": 0}
        dice_state = {class_val: nd_dict.copy() for class_val in self.class_list}
        iou_state = {class_val: nd_dict.copy() for class_val in self.class_list}

        self.metrics_state = {"dice": dice_state, 
                              "iou": iou_state}

    def get_class_masks(self, target, prediction, gt_mask, class_val):
        class_annotation = target == class_val
        class_prediction = (prediction == class_val) & gt_mask
        return class_annotation, class_prediction

    def update_state(self, target, prediction):
        gt_mask = np.isin(target, self.class_list)
        for class_val in self.class_list:
            class_annotation, class_prediction = self.get_class_masks(target, prediction, gt_mask, class_val)

            numer, denom = calculate_iou_nd(class_annotation, class_prediction)
            self.metrics_state['iou'][class_val]['numer'] += numer
            self.metrics_state['iou'][class_val]['denom'] += denom

            numer, denom = caclulate_dice_nd(class_annotation, class_prediction)
            self.metrics_state['dice'][class_val]['numer'] += numer
            self.metrics_state['dice'][class_val]['denom'] += denom
    
    def caclulate_average(self, state_dict):
        class_metrics = []
        for class_val in self.class_list:
            numer = state_dict[class_val]['numer']
            denom = state_dict[class_val]['denom']
            if denom > 0:
                class_metrics.append(numer/denom)
        return float(np.mean(class_metrics))

def read_image(path):
        image = np.array(Image.open(path))
        return image

def evaluate(LocalEvalConfig):
    """
    Runs local evaluation for the model
    Final evaluation code is the same as the evaluator
    """
    datafolder = LocalEvalConfig.DATA_FOLDER
    
    check_data(datafolder)

    imagefolder = os.path.join(datafolder, 'inputs')
    preds_folder = LocalEvalConfig.OUTPUTS_FOLDER

    model = AIcrowdWrapper(predictions_dir=preds_folder, dataset_dir=imagefolder)
    file_names = os.listdir(imagefolder)

    # Predict on all images
    for fname in tqdm(file_names, desc="Predicting Segmentation Masks"):
        model.segment_single_image(fname)

    # Evalaute metrics
    metrics = SegmentationMetrics()
    annotationsfolder = os.path.join(datafolder, 'semantic_annotations')
    for fname in tqdm(file_names, desc="Evaluating results"):
        try:
            semantic_annotation = read_image(os.path.join(annotationsfolder, fname))
            semantic_prediction = read_image(os.path.join(preds_folder, fname))
            metrics.update_state(semantic_annotation, semantic_prediction)
        except ValueError:
            print ("Skip " + fname)

    print("Evaluation Results")

    results = {}
    results['segmentation_mIoU'] = metrics.caclulate_average(metrics.metrics_state['iou'])
    results['segmentation_dice'] = metrics.caclulate_average(metrics.metrics_state['dice'])

    for k,v in results.items():
        print(k,v)


if __name__ == "__main__":
    # change the local config as needed
    class LocalEvalConfig:
        DATA_FOLDER = 'C:\\Users\\Alex\\Downloads\\suadd_23-v0.1'
        OUTPUTS_FOLDER = './evaluator_outputs'

    outfolder=  LocalEvalConfig.OUTPUTS_FOLDER
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    
    evaluate(LocalEvalConfig)