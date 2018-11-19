# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
import argparse
import sys
from extract_cnn_vgg16_keras import VGGNet

os.environ["CUDA_VISIBLE_DEVICES"]="2"
ap = argparse.ArgumentParser()
#ap.add_argument("-database", required = True,
	#help = "Path to database which contains images to be indexed")
ap.add_argument("-index", required = True,
	help = "Name of index file")
args = vars(ap.parse_args())


'''
 Returns a list of filenames for all jpg images in a directory. 
'''
def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]


'''
 Extract features and index the images
'''
if __name__ == "__main__":：

    #db = args["database"]
    img_list =list()
    #aa =['/home/workspace/zhaozhen/data/test123/new']
    db =['/home/data/img59000','/home/data/img100000','/home/data/img1000000',
         '/home/data/img200000','/home/data/img300000','/home/data/img400000',
         '/home/data/img500000','/home/data/img600000','/home/data/img67000',
         '/home/data/img700000','/home/data/img800000','/home/data/img900000']
    for i in range(0,len(db)):
        try:
            #abc = get_imlist(db[i])
            #print type(abc)
            img_list.append(get_imlist(db[i]))
        except:
            continue
    #img = img_list[2]
    #print img
    #print img_list
    print("--------------------------------------------------")
    print("         feature extraction starts")

    print("--------------------------------------------------")
    
    feats = []
    names = []

    model = VGGNet()
    for m in range(0,len(db)):
        imglist = img_list[m]
        #print imglist
        for i, img_path in enumerate(imglist):

            try:
                norm_feat = model.extract_feat(img_path)
                feats.append(norm_feat)
                names.append(img_path)
                print("extracting feature from image No. %d , %d images in total" %((i+1), len(img_list)))
            except:
                continue
    feats = np.array(feats)
    print(feats.shape)

    # directory for storing extracted features
    output = args["index"]
    
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")
    
    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2', data = names)
    h5f.close()
