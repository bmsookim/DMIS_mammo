import cv2
import os

def resizeFiles(file_dir):
    for subdir, dirs, files in os.walk(file_dir):
        for f in files:
            filepath = subdir + os.sep + f
            print filepath

            img = cv2.imread(filepath)
            rsz = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

            os.remove(filepath)
            cv2.imwrite(filepath, rsz)

            print rsz.shape

def checkSize(file_dir):
    for subdir, dirs, files in os.walk(file_dir):
        for f in files:
            filepath = subdir + os.sep + f
            img = cv2.imread(filepath)
            print img.shape

if __name__ == "__main__":
    aug_dir = ["./CALCIFICATION/train/0", "./CALCIFICATION/train/1", "./CALCIFICATION/val/0", "./CALCIFICATION/val/1"]
    for d in aug_dir:
        resizeFiles(d)
    checkSize(aug_dir)
