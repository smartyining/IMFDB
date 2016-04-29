import numpy as np
import pandas as pd
import os

folder_path ='Downloads/IMFDB_final'
allFiles = os.listdir(folder_path)[1:]
imgs = [] # save path to image
labels = [] # save labels

## select a balanced expression data set 
express = ['ANGER','HAPPINESS','SADNESS','SURPRISE','FEAR','DISGUST','NEUTRAL']
express_dict = {}
for i in express:
    express_dict[i] = 0

for f in allFiles:     # is the actor name folder
    subfolders = os.listdir(folder_path+'/'+f)[1:]
    for sf in subfolders:      #sf is the movie name folder
        
        # get .txt file name
        temp = os.listdir(folder_path + '/' + f + '/' + sf)
        for t in temp:
            if t.endswith('.txt'):
                txt_file_name = folder_path + '/' + f + '/' + sf+ '/'+ t
                break
           
        im_base_path = str(folder_path + '/' + f + '/' + sf + '/images')
        # open the .txt file load it into a dataframe
        lines = open(txt_file_name).readlines()   # should be a list
        for l in lines[1:]:
            words = l.split('\t')
            if len(words) == 16:   # there are 16 fields
                im_path = words[1]
                illumination = words[12]
                pose = words[14]
                gender = words[9]    # change here if we want a different target value
                expression = words[10]
                #print(im_path+illumination+pose+gender)

                #if illumination == 'HIGH' and pose == 'FRONTAL':
                if expression in express_dict and express_dict[expression]< 255 and pose == 'FRONTAL' :
                    imgs.append(im_base_path+ '/' +im_path)
                    #labels.append(gender)
                    labels.append(expression)
                    express_dict[expression] += 1
                    
            elif len(words) == 17:  # sometime there might be 17 fields
                im_path = words[2]
                illumination = words[13]
                pose = words[15]
                gender = words[10]
                expression = words[11]                
                #print(im_path+illumination+pose+gender)

                #if illumination == 'HIGH' and pose == 'FRONTAL' and :
                if express_dict[expression]< 255 and pose == 'FRONTAL' :
                    imgs.append(im_base_path+ '/' +im_path)
                    #labels.append(gender)
                    labels.append(expression)
                    express_dict[expression] += 1
                    
                    
output_file = open('bf_expression','w')
for i in range(len(imgs)):
    output_file.write(imgs[i] + ',' + labels[i]+'\n')
output_file.close()
    
               