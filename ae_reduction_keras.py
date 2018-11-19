from autoencoder_k import autoencoder_model
from numpy as np
import argparse
import os
import h5py

os.environ["CUDA_VISIBLE_DEVICES"]="3"

ap = argparse.ArgumentParser()
ap.add_argument("-feats",required=True,
                help="feats prepared to reduct")
ap.add_argument("-rfeats",required=True,
                help="feats reducted")
args = vars(ap.parse_args())

h5f = h5py.File(args["feats"],'r')

feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]

autoencoder_model(feats)

output = args["rfeats"]
h5f = h5py.File(output,'w')
h5f.create_dataset('dataset_1', data=feats)
h5f.create_dataset('dataset_2', data=imgNames)

h5f.close()