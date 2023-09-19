import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2
import PIL.Image
  
from sklearn.model_selection import train_test_split
from labelme import utils


class Labelme2YOLO(object):
    def __init__(self, _json_dir):
        self._json_dir = _json_dir
        
        self._label_id_map = self._get_label_id_map(self._json_dir)
        
    def _make_train_val_test_dir(self):
        for yolo_path in ('train', 'valid', 'train/images', 'train/labels', 'valid/images', 'valid/labels'):
            yolo_path = os.path.join(self._json_dir, 'YOLODataset', yolo_path)
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)
            os.makedirs(yolo_path)
                
    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        
        return OrderedDict([(label, label_id) for label_id, label in enumerate(label_set)])
    
    def _train_test_split(self, json_names, val_size):
        train_idxs, val_idxs = train_test_split(range(len(json_names)), test_size=val_size)
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        
        return train_json_names, val_json_names
    
    def convert(self, val_size):
        # all the JSON files listed
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders =  [file_name for file_name in os.listdir(self._json_dir) \
                    if os.path.isdir(os.path.join(self._json_dir, file_name))]
        train_json_names, val_json_names = self._train_test_split(json_names, val_size)
        
        self._make_train_val_test_dir()
    
        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_names in zip(('train', 'valid'), 
                                          (train_json_names, val_json_names)):
            for json_name in json_names:
                json_path = os.path.join(self._json_dir, json_name)
                json_data = json.load(open(json_path))
                
                print('Converting %s for %s ...' % (json_name, target_dir.replace('/', '')))
                
                self._save_yolo_image(json_name, target_dir)
                    
                yolo_obj_list = self._get_yolo_object_list(json_data)
                self._save_yolo_label(json_name, target_dir, yolo_obj_list)
        
        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()
                

    
    def _get_yolo_object_list(self, json_data):
        yolo_obj_list = []
        for shape in json_data['shapes'] :
            label_id = self._label_id_map[shape['label']]
            yolo_obj_line = str(label_id)
            for point in shape['points'] : 
                yolo_obj_line += " " + str(point[0]) + " " + str(point[1])
            yolo_obj_list.append(yolo_obj_line)
        return yolo_obj_list

    
    def _save_yolo_label(self, json_name, target_dir, yolo_obj_list):
        txt_path = os.path.join(self._json_dir, 'YOLODataset', target_dir, 'labels', json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj_line in enumerate(yolo_obj_list):
                if yolo_obj_idx + 1 != len(yolo_obj_list) :
                    yolo_obj_line += "\n"
                f.write(yolo_obj_line)
                
    def _save_yolo_image(self, json_name, target_dir):
        img_name = json_name.replace('.json', '.jpg')
        img_path = os.path.join(self._json_dir, img_name)
        new_img_path = os.path.join(self._json_dir, 'YOLODataset', target_dir, 'images', img_name)
        print(img_path)
        image = cv2.imread(img_path)
        cv2.imwrite(new_img_path, image)
    
    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._json_dir, 'YOLODataset', 'dataset.yaml')
        
        with open(yaml_path, 'w+') as yaml_file:
            train_dir = os.path.join(self._json_dir, 'YOLODataset', "train")
            val_dir = os.path.join(self._json_dir, 'YOLODataset', "valid")
            yaml_file.write('train: %s\n' % \
                            train_dir)
            yaml_file.write('val: %s\n\n' % \
                            val_dir)
            yaml_file.write('nc: %i\n\n' % len(self._label_id_map))
            
            names_str = ''
            for label, _ in self._label_id_map.items():
                names_str += "'%s', " % label
            names_str = names_str.rstrip(', ')
            yaml_file.write('names: [%s]' % names_str)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir',type=str,
                        help='Please input the path of the labelme json files.')
    parser.add_argument('--val_size',type=float, nargs='?', default=None,
                        help='Please input the validation dataset size, for example 0.1 ')
    args = parser.parse_args(sys.argv[1:])
         
    convertor = Labelme2YOLO(args.json_dir)
    convertor.convert(val_size=args.val_size)

    
