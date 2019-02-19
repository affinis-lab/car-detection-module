import json


class GroundTruth:
    """
    this class loads the ground truth
    """
    def __init__(self, config):
        self._objects_all = []  # positive bboxes across images
        self.config = config
        self.annotations_file = config['data']['annotations_file']
        self.images = config['data']['images']

    def update_objects_all(self, dictObj):
        self._objects_all.append(dictObj)

    def load_json(self):
        with open(self.annotations_file) as f:
            data = json.load(f)

        for key in data:
            objects_per_image = list()
            dictObj = dict()
            dictObj['filename'] = self.images + data[key]['filename']

            regions = data[key]['regions']

            for region in regions:
                readObject = dict()
                readObject['name'] = 'car'
                readObject['xmin'] = float(region['shape_attributes']['x'])
                readObject['ymin'] = float(region['shape_attributes']['y'])
                readObject['xmax'] = float(region['shape_attributes']['x']) + float(region['shape_attributes']['width'])
                readObject['ymax'] = float(region['shape_attributes']['y']) + float(region['shape_attributes']['height'])
                objects_per_image.append(readObject)

            dictObj['object'] = objects_per_image
            dictObj['width'] = self.config['data']['images_width']
            dictObj['height'] = self.config['data']['images_height']
            self.update_objects_all(dictObj)

    def objects_all(self):
        return self._objects_all
