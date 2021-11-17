"""

MAIN MODULE

-Calls module to block the mzml file into images
-Conducts yolov5 inference using the command line and saves the predictions to txt and image files
-Converts inference .txt files into m/z and rt + ranges CSV file

Computer Specifications:

Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
NVIDIA GeForce GTX 1070
32 GB 2133mHz RAM
500GB Samsung SSD

"""

import os
from get_object_block_imgs import get_image_for_blocks
#Import debug module
from get_object_block_training_and_debug import extract_training_data, check_similarities_between_v1_v2
import sys
#TODO This doesn't work for me because Xiuxia's forked repository is ADAP-3D_v2 instead of ADAP-3D-V2
# To avoid using `sys.path.append(r"../ADAP-3D-V2/yolov5")`, just call `from yolov5.detect import run` instead.
# If the latter doesn't work, add __init__.py into the folder yolov5
sys.path.append(r"../ADAP-3D-V2/yolov5")
from detect import run
from convert_txt_to_mz_rt import getinferencearrs, getinferencefilenames, convert
import params
import pickle
import gc
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np

def main():
    #TODO This will break if multiple users would try to run it. Can we move "Blocks" to a temporary folder?
    # Temporary folder is a cleaner approach in general. See https://docs.python.org/3/library/tempfile.html
    if not os.path.isdir(r'../Blocks'):
      os.system('mkdir .\Blocks')
    extract_training_data(params.profile_mzml_path, window_mz = 48, window_rt= 80)

    imgs = get_image_for_blocks(params.profile_mzml_path, window_mz = 48, window_rt= 80, timetoignoreL = 2.5, timetoignoreR = 19, min_intensity_threshold = 1000)
    #pickle.dump(imgs, open(params.results_path + "\\img_data.p", "wb"))
    #imgs = pickle.load(open(params.results_path + "\\img_data.p", 'rb'))

    run(weights=params.weights_path, imgsz=160, conf_thres=0.2, source = params.source_path, save_txt=True, save_conf=True)
    arrofpredictions = getinferencearrs()
    arroffilenames = getinferencefilenames()
    dataframetoexport = convert(imgs[2], imgs[1], arroffilenames, arrofpredictions, params.window_mz, params.window_rt)
    dataframetoexport.to_csv(params.results_path + "\Final Predictions.csv")

#TODO Use package argparse to add command-line arguments to select a raw data file and other options if necessary.
if __name__ == '__main__':
    main()

