import argparse
import json


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-coco', '--coco-file', required=True, type=str)
    parser.add_argument('-out', '--out-file', required=True, type=str)
    return parser


def get_new_categories():
    new_categories = [{"supercategory": "none", "id": 1, "name": "Person"},
                      {"supercategory": "none", "id": 2, "name": "Car"}]
    old_category_id_to_new = {14: 2, 24: 1}
    return new_categories, old_category_id_to_new


def get_new_annotations(annotations, old_category_id_to_new, start_id=1):
    new_annotations = list()
    used_images_id = set()
    idx = start_id
    for annotation in annotations:
        if annotation['category_id'] not in old_category_id_to_new.keys():
            continue
        annotation['category_id'] = old_category_id_to_new[annotation['category_id']]
        annotation['id'] = idx
        new_annotations.append(annotation)
        used_images_id.add(annotation['image_id'])
        idx += 1
    return new_annotations, used_images_id


def get_new_images(images, used_images_id, start_id=0):
    new_images = list()
    old_image_id_to_new = dict()
    idx = start_id
    for image in images:
        if image['id'] not in used_images_id:
            continue
        old_image_id_to_new[image['id']] = idx
        image['id'] = idx
        new_images.append(image)
        idx += 1
    return new_images, old_image_id_to_new


def remove_classes(coco_file, out_file):
    with open(coco_file, 'r') as f:
        json_dict = json.load(f)
    images = json_dict['images']
    annotations = json_dict['annotations']

    categories, old_category_id_to_new = get_new_categories()
    annotations, used_images_id = get_new_annotations(annotations, old_category_id_to_new)
    images, old_image_id_to_new = get_new_images(images, used_images_id)
    for annotation in annotations:
        annotation['image_id'] = old_image_id_to_new[annotation['image_id']]

    json_dict = {'images': images, 'annotations': annotations, 'categories': categories}
    with open(out_file, 'w') as f:
        json.dump(json_dict, f)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    remove_classes(**vars(args))
