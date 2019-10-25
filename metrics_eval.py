import argparse
from instance_eval import detection_metrics
from instance_eval import Params
from pycocotools.coco import COCO
import json


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ann', '--annotations_file', required=True, type=str)
    parser.add_argument('-det', '--detections_file', required=True, type=str)
    return parser


def evaluate_detections(annotations_file, detections_file):
    coco_gt = COCO(annotations_file)
    with open(detections_file) as f:
        dt_json = json.load(f)
    coco_dt = coco_gt.loadRes(dt_json)
    params = Params(coco_gt, iouType='bbox')
    metrics = detection_metrics(coco_gt, coco_dt, params)
    return metrics


def extract_mAP(metrics, iouThrs=0.5):
    permitted_iouThrs = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    iouThrs_type = type(iouThrs)
    if iouThrs_type in (float, int):
        iouThrs = (iouThrs,)
    for iouThr in iouThrs:
        assert iouThr in permitted_iouThrs

    indexed = metrics.set_index(["area", "maxDet"])
    area = 'custom'
    maxDet = 100
    mAPs = []
    for iouThr in iouThrs:
        mAP = indexed.loc[(area, maxDet)].reset_index().set_index(["area", "maxDet", "iouThr"]).loc[(area, maxDet, iouThr)]["AP"].mean()
        mAPs.append(mAP)

    if iouThrs_type in (float, int):
        mAPs = mAPs[0]
    return mAPs


def extract_AP(metrics, classes, iouThrs=0.5):
    iouThrs_type = type(iouThrs)
    if iouThrs_type in (float, int):
        iouThrs = (iouThrs,)
    classes_type = type(classes)
    if classes_type in (str,):
        classes = (classes,)

    permitted_iouThrs = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    for iouThr in iouThrs:
        assert iouThr in permitted_iouThrs

    APs = []
    area = 'custom'
    maxDet = 100
    for iouThr in iouThrs:
        APs_iouThr = []
        for cl in classes:
            cl_idxes = [idx for idx, value in enumerate(metrics['class']) if value == cl]
            area_idxes = [idx for idx, value in enumerate(metrics['area']) if value == area and idx in cl_idxes]
            maxDet_idxes = [idx for idx, value in enumerate(metrics['maxDet']) if value == maxDet and idx in area_idxes]
            iouThr_idxes = [idx for idx, value in enumerate(metrics['iouThr']) if value == iouThr and idx in maxDet_idxes]
            assert len(iouThr_idxes) == 1
            idx = iouThr_idxes[0]
            APs_iouThr.append(metrics['AP'][idx])
        APs.append(APs_iouThr)

    if iouThrs_type in (float, int):
        APs = APs[0]
    if classes_type in (str,):
        if iouThrs_type in (float, int):
            APs = APs[0]
        else:
            APs = [APs[i][0] for i in range(len(APs))]
    return APs


def get_optimal_score_threshold(metrics, classes, iouThrs=0.5):
    iouThrs_type = type(iouThrs)
    if iouThrs_type in (float, int):
        iouThrs = (iouThrs,)
    classes_type = type(classes)
    if classes_type in (str,):
        classes = (classes,)

    permitted_iouThrs = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    for iouThr in iouThrs:
        assert iouThr in permitted_iouThrs

    class gt:
        def __init__(self):
            self.cats = dict()
        def getImgIds(self):
            return [0]
    metrics_thrs = Params(gt(), iouType='bbox').recThrs

    opt_thrs = []
    area = 'custom'
    maxDet = 100
    for iouThr in iouThrs:
        opt_thrs_iouThr = []
        for cl in classes:
            cl_idxes = [idx for idx, value in enumerate(metrics['class']) if value == cl]
            area_idxes = [idx for idx, value in enumerate(metrics['area']) if value == area and idx in cl_idxes]
            maxDet_idxes = [idx for idx, value in enumerate(metrics['maxDet']) if value == maxDet and idx in area_idxes]
            iouThr_idxes = [idx for idx, value in enumerate(metrics['iouThr']) if
                            value == iouThr and idx in maxDet_idxes]
            assert len(iouThr_idxes) == 1
            idx = iouThr_idxes[0]
            diff = 2  # >1
            opt_thr = -1
            for i in range(len(metrics['recall'][idx])):
                new_diff = abs(metrics['recall'][idx][i] - metrics['precision'][idx][i])
                if new_diff < diff:
                    diff = new_diff
                    opt_thr = metrics_thrs[i]
            assert opt_thr != -1
            opt_thrs_iouThr.append(opt_thr)
        opt_thrs.append(opt_thrs_iouThr)

    if iouThrs_type in (float, int):
        opt_thrs = opt_thrs[0]
    if classes_type in (str,):
        if iouThrs_type in (float, int):
            opt_thrs = opt_thrs[0]
        else:
            opt_thrs = [opt_thrs[i][0] for i in range(len(opt_thrs))]
    return opt_thrs


def main(annotations_file, detections_file):
    classes = ('Person', 'Car')
    metrics = evaluate_detections(annotations_file, detections_file)
    AP = extract_AP(metrics, classes)
    mAP = extract_mAP(metrics, iouThrs=0.5)
    for i in range(len(classes)):
        print('{:15} {}'.format(classes[i], AP[i]))
    print('')
    print('{:15} {}'.format('mAP', mAP))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(**vars(args))
