# Import python modules
import cv2
import os
import numpy as np
import os.path

# Define directories
input_path = './CALCIFICATION/train/'

def getGlobalMean(in_dir):
    count = 0
    mean = 0
    for subdir, dirs, files in os.walk(in_dir):
        for f in files:
            filepath = subdir + os.sep + f

            if filepath.endswith(".jpg") or filepath.endswith(".png"):
                print "Processing " + filepath + "..."
                count += 1
                img = cv2.imread(filepath)
                mean += img.mean()

    return (mean / count)

if __name__ == "__main__":
    global_mean = getGlobalMean(input_path)
    print float(global_mean)/255.0
