import os
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import cv2

from decode_np import Decode


def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

if __name__ == '__main__':
    print('Please visit https://github.com/miemie2013/Keras-YOLOv4 for more complete model!')

    # #model_path = 'yolo4_PPEweight.h5'
    # anchors_path = 'model_data/anchors_PPE.txt'
    # #classes_path = 'model_data/PPE_classes.txt'
    # model_path = 'yolov4-helmet-detection.h5'
    # classes_path = 'model_data/person_helmet_head_classes.txt'

    # mask detection
    # model_path = 'yolo4_maskweight.h5'
    # anchors_path = 'model_data/anchors_PPE.txt'
    # classes_path = 'model_data/mask_classes.txt'


    ## gloves_glasses_detection
    # model_path = 'yolov4-gloves_glasses-detection.h5'
    # anchors_path = 'model_data/anchors_PPE.txt'
    # classes_path = 'model_data/gloves_glasses_classes.txt'

    # ##crowd_person_detection
    # model_path = 'yolov4-crowdperson-detection.h5'
    # anchors_path = 'model_data/anchors_PPE.txt'
    # classes_path = 'model_data/crowdperson_classes.txt'

    ##person_helmet_mask_detection
    model_path = 'yolov4-person_helmet_mask-detection.h5'
    anchors_path = 'model_data/anchors_PPE.txt'
    classes_path = 'model_data/person_helmet_mask_classes.txt'

    ##person_helmet_mask_detection
    model_path = 'yolov4_80_classes.h5'
    anchors_path = 'model_data/anchors_PPE.txt'
    classes_path = 'model_data/coco_classes.txt'


    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)                      

    model_image_size = (416, 416)


    conf_thresh = 0.2
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)
    video_path = "D:/deeplearning learn/marico_demo/airport803_18-43-22.avi"
    ##video_path = "D:/deeplearning learn/marico_demo/videos for aseptic practices/3. currect practice_body and hand position.MP4"
    _decode.detect_video(video_path)

    ## for image
    # while True:
    #     img = "D:/deeplearning learn/marico_demo/kodikal/IMG/image_1500.jpg"
    #     try:
    #         image = cv2.imread(img)
    #     except:
    #         print('Open Error! Try again!')
    #         continue
    #     else:
    #         image, boxes, scores, classes = _decode.detect_image(image, True)
    #         cv2.imshow('image', image)
    #         cv2.imwrite("image_1500.jpg",image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

    yolo4_model.close_session()