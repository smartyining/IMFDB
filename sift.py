# get sift feature

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def sift(infile, vis=False):
    result = []
    labels = []
    sift = cv2.SIFT()
    ix = 0
    
    for here, i in enumerate(open(infile).readlines()):
        ix +=1
        imgpath, l = i.split(',')
        try:
            img = cv2.imread(imgpath)        
            kp, des = sift.detectAndCompute(img,None)

            if vis == True:
                plt.figure(figsize=(10,16))
                img_kp = cv2.drawKeypoints(img, kp, color=(255,0,0))
                plt.subplot(2, 5, ix)
                plt.imshow(img_kp)        
                #visulization
            
            if des!=[]:
                result.append(des)
                labels.append(l)            
        except:
            pass

    print len(result)
    print len(labels)

    return result,labels

def main():
    result, labels = sift('all_gender')

    f1= open('keypoints_sift_all.txt','w')
    f2 = open('label_all.txt','w')
    ix = 0
    for kps in result:
        # for each img
        ix +=1
        if kps!= None:
            for kp in kps:
                f1.write(str(ix)+',')
                for i in kp:
                    f1.write(str(i)+' ')
                f1.write('\n')
            f2.write(str(ix)+','+labels[ix-1])
    f1.close()
    f2.close() 

if __name__ == "__main__":
    main()






     