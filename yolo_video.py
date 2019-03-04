import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import time
import scipy.io as scio
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.save('yoloimage_ori.jpg')
    yolo.close_session()

def detect_img_pairs(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image,yoloopt,details = yolo.detect_image_pairs(image)
            ticks = time.time()
            dataNew='./yolo_outputs'+str(ticks)+'.mat'
            #sess=tf.Session()
            for i in range(len(details)):
                details[i]['box_xy']=np.float32(details[i]['box_xy'])
                details[i]['box_wh']=np.float32(details[i]['box_wh'])
                details[i]['box_confidence']=np.float32(details[i]['box_confidence'])
                details[i]['feats']=np.float32(details[i]['feats'])
            scio.savemat(dataNew, {'yolo_outputs0':np.float32(yoloopt[0]),'yolo_outputs1':np.float32(yoloopt[1]),'yolo_outputs2':np.float32(yoloopt[2]),'details0':details[0],'details1':details[1],'details2':details[2]})
            #r_image.show()
            r_image.save('1234.jpg')
            print(yoloopt[0][0][0][0])
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    parser.add_argument(
        '--pairs', default=False, action="store_true",
        help='Pairs model mode, its pairs detect head and body'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        if FLAGS.pairs:
            print('now is pairs mode:')
            detect_img_pairs(YOLO(**vars(FLAGS)))
            
        else:
            detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
