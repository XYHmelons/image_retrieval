# -*- coding: utf-8 -*-
from extract_cnn_vgg16_keras import VGGNet
import time
import numpy
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os
from sklearn.externals import joblib
from sklearn.decomposition import RandomizedPCA
from autoencoder_k import use_ae_reduction


os.environ["CUDA_VISIBLE_DEVICES"]="1"

ap = argparse.ArgumentParser()
ap.add_argument("-query", required=True,
                help="Path to query which contains image to be queried")
ap.add_argument("-index", required=True,
                help="Path to index")
# ap.add_argument("-result", required = True,
# help = "Path for output retrieved images")
args = vars(ap.parse_args())
#--------------------------------------
# read in indexed images' feature vectors and corresponding image names加载数据
h5f = h5py.File(args["index"], 'r')

feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
print(feats.shape)
print(feats[0:15])

print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
queryDir = args["query"]
#queryImg = mpimg.imread(queryDir)  # 读取图片
# plt.title("Query Image")
# plt.imshow(queryImg)
# plt.show()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
print(queryVec.shape)
queryVec =  use_ae_reduction(queryVec)

starttime = time.time()
'''-------------------------设置分割个数--------------------------'''
scores_list =[]
rank_ID = []
rank_scores = []
batch_size = 5
length = len(feats)
split = length//batch_size
print(split)
#print(feats[length]==feats[length])

'''--------------------------------------------------------------'''
for i in range(batch_size):
    start = (i*split) % length
    end = min(start+split , length)

    feat = feats[start:end]
    scores = np.dot(queryVec, feat.T)
    print('*********************')
    print(feat.shape)
    print(scores.shape)
    print(scores)
    scores_list.extend(scores)
    #scores_list = np.array(scores_list)
    #rank_ID1 = np.argsort(scores)[::-1]
    #rank_score1 = scores[rank_ID1]
    #print rank_score1
    #print rank_ID1
    #rank_ID.extend(rank_ID1[0:1000])
    #rank_scores.extend(rank_score1[0:1000])

if end != length:
    feat = feats[end:length]
    scores = np.dot(queryVec, feat.T)
    print('*********************')
    scores_list.extend(scores)                            #分数列表
    scores_list = np.array(scores_list)                   #将列表转换为数组
    #rank_ID1 = np.argsort(scores)[::-1]
    #rank_score1 = scores[rank_ID1]
    #rank_ID.extend(rank_ID1)
    #rank_scores.extend(rank_score1)
    #rank_scores = np.array(rank_scores)
    #print rank_scores
   # rank_ID=np.array(rank_ID)
   # print rank_ID
    #rank_ID = np.argsort(rank_scores)[::-1]
    #print scores_list.shape
else:
    scores_list=np.array(scores_list)
    print('123')

rank_ID = np.argsort(scores_list)[::-1]  # 并将所有目标框按照score重新逆序排列 （argsort默认设置为标签从低到高）
print('#########33###############')
print(rank_ID)
print(rank_ID.shape)
#rank_score = scores_list[rank_ID]
#print rank_score
#print rank_score.shape

# number of top retrieved images to show
maxres = 100
imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
endtime = time.time()
time_cost = endtime - starttime
print("top %d images in order are: " % maxres, imlist)
print('Time cost:{}...'.format(time_cost))

# show top #maxres retrieved result one by one
for i, im in enumerate(imlist):
    try:

        # image = mpimg.imread(im)
        img = Image.open(im)
        img.save('/home/workspace/xieyunhao/project/image_retrieval/Cosine_distance/retrieval/%s.jpg' % i)
        # plt.title("search output %d" % (i + 1))
        # plt.imshow(image)
        # plt.show()
    except:
        continue