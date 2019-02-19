import os
import cv2
import copy
import numpy as np
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder

from utils import bbox_iou

GRID_H = 13
GRID_W = 13
BOX = 2
BATCH_SIZE = 16
IMAGE_SIZE = 416
TRUE_BOX_BUFFER = 50

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}

        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels

class BatchGenerator(Sequence):
    def __init__(self, dictionaries,
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None
        self.labels = ['car']
        self.dictionaries = dictionaries
        self.config = config
        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm
        self.n_channels = 3
        self.max_obj = 7
        self.nb_anchors = 2
        self.anchors_constants = [0.65,1.33, 2.26,4.31]
        self.anchors = [[0, 0, self.anchors_constants[2 * i], self.anchors_constants[2 * i + 1]] for i in
         range(int(len(self.anchors_constants) // 2))]

        if shuffle: np.random.shuffle(self.dictionaries)

    def __len__(self):
        return int(np.ceil(float(len(self.dictionaries))/BATCH_SIZE))

    def num_classes(self):
        return 1

    def size(self):
        return len(self.dictionaries)

    def load_annotation(self, i):
        annots = []

        for obj in self.dictionaries[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.dictionaries[i]['filename'])

    def __getitem__(self, idx):
        le = LabelEncoder()
        le.fit_transform(self.labels)

        x_batch = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, self.n_channels))
        b_batch = np.zeros((BATCH_SIZE, 1, 1, 1, self.max_obj, 4))

        y_batch = np.zeros((BATCH_SIZE, GRID_H, GRID_W, self.nb_anchors,
                            4 + 1 + self.num_classes()))  # desired network output

        # current_batch = self.dataset[l_bound:r_bound]
        current_batch = self.dictionaries[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        instance_num = 0

        for instance in current_batch:
            img, object_annotations = self.aug_image(instance, jitter=self.jitter)

            obj_num = 0

            # center of the bounding box is divided with the image width/height and grid width/height
            # to get the coordinates relative to a single element of a grid
            for obj in object_annotations:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.labels:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])  # center of the lower side of the bb (by x axis)
                    center_x = center_x / (
                    float(IMAGE_SIZE) / GRID_W)  # scaled to the grid unit (a value between 0 and GRID_W-1)
                    center_y = .5 * (obj['ymin'] + obj['ymax'])  # center of the lower side (by y axis)
                    center_y = center_y / (
                    float(IMAGE_SIZE) / GRID_H)  # scaled to the grid unit (a value between 0 and GRID_H-1)

                    grid_x = int(np.floor(center_x))  # assigns the object to the matching
                    grid_y = int(np.floor(center_y))  # grid element according to (center_x, center_y)

                    if grid_x < GRID_W and grid_y < GRID_H:
                        center_w = (obj['xmax'] - obj['xmin']) / (float(IMAGE_SIZE) / GRID_W)
                        center_h = (obj['ymax'] - obj['ymin']) / (float(IMAGE_SIZE) / GRID_H)

                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = [0, 0, center_w, center_h]

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        img = self.normalize(img)

                        x_batch[instance_num] = img

                        b_batch[instance_num, 0, 0, 0, obj_num] = box
                        y_batch[instance_num, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_num, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_num, grid_y, grid_x, best_anchor, 5] = 1

                        obj_num += 1
                        obj_num %= self.max_obj

            instance_num += 1

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.dictionaries)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy : (offy + h), offx : (offx + w)]

    #         ### flip the image
    #         flip = np.random.binomial(1, .5)
    #         if flip > 0.5: image = cv2.flip(image, 1)
    #
    #         image = self.aug_pipe.augment_image(image)
    #
        # resize the image to standard size
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(IMAGE_SIZE) / w)
                obj[attr] = max(min(obj[attr], IMAGE_SIZE), 0)

            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(IMAGE_SIZE) / h)
                obj[attr] = max(min(obj[attr], IMAGE_SIZE), 0)

    #         if jitter and flip > 0.5:
    #             xmin = obj['xmin']
    #             obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
    #             obj['xmax'] = self.config['IMAGE_W'] - xmin
    #
        return image, all_objs

    def normalize(self, image):
        return image / 255.
