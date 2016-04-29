# get HOG feature
# this run HOG feauture extraction and run svm classifier

import cv2
import os
import numpy as np
from sklearn import svm

def hog(infile):
    result = []   
    label = []
    hog = cv2.HOGDescriptor()
    ix = 0

    for here, i in enumerate(open(infile).readlines()):
        ix +=1
        imgpath, l = i.split(',')
        if os.path.exists(imgpath):

            img = cv2.imread(imgpath)
            img  = cv2.resize(img, (200, 200))
            des = hog.compute(img)
            if des.ndim ==2:
                result.append(map(lambda x:float(x),des))
                if "FE" in l:
                    label.append(1) 
                else:
                    label.append(-1) 
                
    print '%d,%d'%(len(result),len(label))
    return result,label


# from sklearn import svm

def main():
    clf = svm.SVC()
    clf.fit(result[:600], label[:600]) 
    print clf.score(result[600:], label[600:])


if __name__ == "__main__":
    main()




