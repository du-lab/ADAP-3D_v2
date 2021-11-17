"""

PARAMETERS MODULE

-Set parameters here
-May change to OOP organization if needed

"""

#TODO Change to OOP. Don't use any absolute paths here. They won't work on my computer.

profile_mzml_path = r"C:\Users\jerry\Desktop\DCSM_PROFILE.mzML" #File path to the mzML file (profiled form only) to conduct peak detection on
#profile_mzml_path = r"C:\Users\jerry\Desktop\VT001.mzml" #File path to the mzML file (profiled form only) to conduct peak detection on
#profile_mzml_path = r"C:\Users\jerry\Desktop\YP01 (1).cdf" #File path to the mzML file (profiled form only) to conduct peak detection on
results_path = r"C:\Users\jerry\Desktop\Results" #File path to results folder
weights_path = "C:\\Users\\jerry\\Desktop\\ADAP-3D-V2\\src\\final_pre_trained_model.pt" #File path to weights
source_path = "C:\\Users\\jerry\\Desktop\\ADAP-3D-V2\\Blocks" #File path to the source of images used for object detection
txt_inference_path = "C:\\Users\\jerry\\Desktop\\ADAP-3D-V2\\yolov5\\runs\\detect\\exp4\\labels" #File path to inferred txt files containing bounding boxes
window_mz = 48 # Number of mz values * 2 = displayed in image
window_rt = 80 # Number of time values * 2 = displayed in image
