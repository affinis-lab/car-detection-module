from keras.models import load_model
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from tiny_yolo import TinyYolo
from utils import decode_netout, draw_boxes

CONFIG_FILE = 'config.json'


def main():

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    yolo = TinyYolo(input_size=416, config=config)

    if config['test_config']['training']:
        yolo.train()
    else:
        yolo.model = load_model('weights_coco.h5', custom_objects={'custom_loss': yolo.custom_loss})

        yolo.model.summary()

        for img in config['test_config']["test_images"]:
            image = cv2.imread(config['test_config']['test_images_path'] + img)

            plt.figure(figsize=(10, 10))

            input_image = cv2.resize(image, (416, 416))
            input_image = input_image / 255.
            input_image = np.expand_dims(input_image, 0)

            dummy_array = np.zeros((1, 1, 1, 1, config['model']['max_obj'], 4))

            start = time.time()
            netout = yolo.model.predict([input_image, dummy_array])[0]
            end = time.time()

            print("Prediction took " + str(end - start) + " seconds.")

            boxes = decode_netout(netout,
                                  obj_threshold=config['test_config']['obj_threshold'],
                                  nms_threshold=config['test_config']['nms_threshold'],
                                  anchors=config['model']['anchors'],
                                  nb_class=config['model']['nb_class'])

            # for box in boxes:
            #     print(box.xmin, box.ymin, box.xmax, box.ymax, box.score)
            image = draw_boxes(image, boxes, labels=["car"])

            plt.imshow(image[:, :, ::-1])
            plt.show()

if __name__ == '__main__':
    main()