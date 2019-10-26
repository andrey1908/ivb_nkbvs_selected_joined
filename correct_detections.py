import argparse
import json


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-coco', '--coco-file', required=True, type=str)
    parser.add_argument('-out', '--out-file', required=True, type=str)
    return parser


def correct_detections(coco_file, out_file):
    with open(coco_file, 'r') as f:
        detections = json.load(f)

    old_category_id_to_new = {1: 1, 3: 2, 4: 2, 6: 2}
    new_detections = list()
    for detection in detections:
        if detection['category_id'] not in old_category_id_to_new.keys():
            continue
        detection['category_id'] = old_category_id_to_new[detection['category_id']]
        new_detections.append(detection)

    with open(out_file, 'w') as f:
        json.dump(new_detections, f)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    correct_detections(**vars(args))
