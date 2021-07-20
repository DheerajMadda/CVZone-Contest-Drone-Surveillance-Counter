import sys
sys.path.insert(0, './packages')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams

class Yolov5():
    def __init__(self, weights, device):
        self.device = device
        self.model = attempt_load(weights, map_location=device)

    def load(self):
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        return self.model

    def load_images(self, path):
        return LoadImages(path)


