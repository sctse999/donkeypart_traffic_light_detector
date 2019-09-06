import numpy as np
import cv2
import time
import random
import collections
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from matplotlib import cm


class TrafficLightDetector(object):
    '''
    Requires an EdgeTPU for this part to work

    This part will run a EdgeTPU optimized model to run object detection to detect a traffic light.
    We are just using a pre-trained model (MobileNet V2 SSD) provided by Google.

    After we identified the


    We override the throttle


    Detect the finishing line of a track. Assuming the finishing line is red and if the car
    pass the finish line, the lower portion of the frame should contain a certain percentage of red color

    Use opencv to find out the percentage of red color covering the lower portion of the frame and return
    whether the car has passed the finishing line

    '''

    # MODEL_URL = "https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
    # LABEL_URL =

    def __init__(self):
        model = "/home/jonathantse/Downloads/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
        label = "/home/jonathantse/Downloads/coco_labels.txt"
        self.last_5_scores = collections.deque(np.zeros(5), maxlen=5)
        self.engine = DetectionEngine(model)
        self.labels = dataset_utils.ReadLabelFile(label)

        self.TRAFFIC_LIGHT_CLASS = 9
        self.LAST_5_SCORE_THRESHOLD = 0.4
        self.MIN_SCORE = 0.2

    def convertImageArrayToPILImage(self, img_arr):
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')

        # img = Image.fromarray(np.uint8(cm.gist_earth(img_arr)*255))
        return img

    '''
    Return an object if there is a traffic light in the frame
    '''
    def detect_traffic_light(self, img_arr):
        img = self.convertImageArrayToPILImage(img_arr)

        ans = self.engine.DetectWithImage(img,
                                          threshold=self.MIN_SCORE,
                                          keep_aspect_ratio=False,
                                          relative_coord=False,
                                          top_k=3)

        max_score = 0
        traffic_light_obj = None
        if ans:
            for obj in ans:
                if (obj.label_id == self.TRAFFIC_LIGHT_CLASS):
                    if (obj.score > max_score):
                        traffic_light_obj = obj
                        max_score = obj.score

        if traffic_light_obj:
            self.last_5_scores.append(traffic_light_obj.score)
            sum_of_last_5_score = sum(list(self.last_5_scores))

            if sum_of_last_5_score > self.LAST_5_SCORE_THRESHOLD:
                return traffic_light_obj
            else:
                print("Not reaching last 5 score threshold")
                return None
        else:
            return None

        return traffic_light_obj

    '''
    Return a traffic light image array based on the traffic light obj bounding box
    '''
    def crop_traffic_light(self, img_arr, traffic_light_obj):
        bbox = traffic_light_obj.bounding_box

        x1 = int(bbox[0][0])
        x2 = int(bbox[1][0])

        y1 = int(bbox[0][1])
        y2 = int(bbox[1][1])

        return img_arr[y1:y2, x1:x2]

    def is_light_on(self, img_arr):
        img_gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        ret, img_thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(img_thresh) > 0:
            return True
        else:
            return False

    def crop_upper_half(self, img_arr):
        h, w, channels = img_arr.shape
        upper_half_img_arr = img_arr[
            0:h // 2, 0:w, :]  # Upper part of image refer to red light

        return upper_half_img_arr

    def run(self, img_arr, throttle, debug=False):
        if img_arr is None:
            return False, img_arr

        if debug:
            cv2.imshow("img {}".format(random.randint(1, 10000)), img_arr)
        #     cv2.waitKey()

        # Detect traffic light object
        traffic_light_obj = self.detect_traffic_light(img_arr)

        if traffic_light_obj:
            print(traffic_light_obj.score)
            cv2.rectangle(img_arr, tuple(traffic_light_obj.bounding_box[0].astype(int)),
                          tuple(traffic_light_obj.bounding_box[1].astype(int)), (0, 255, 0), 2)

            traffic_light_img = self.crop_traffic_light(
                img_arr, traffic_light_obj)

            upper_half_img_arr = self.crop_upper_half(img_arr)

            if self.is_light_on(upper_half_img_arr):
                throttle = 0

        return throttle, img_arr

    # def shutdown(self):
    # if self.fps_list is not None:
    #     print("fps (min/max) = {:2d} / {:2d}".format(min(self.fps_list), max(self.fps_list)))
    #     print("fps list {}".format(self.fps_list))
    # self.running = False
    # time.sleep(0.1)
