import os
import pafy
import numpy as np
import urllib
#import urllib2
from moviepy.editor import *
from scipy import stats
import sqlite3
import csv
import h5py
import scipy.misc
import shutil
import imageio

def preprocess_video(query,videoURL):
    
    Gquery=query
    # video = pafy.new(videoURL)
    # for s in video.streams:
    #     best = s
        
    direc = "/content/drive/MyDrive/New_Code/New model/videos"
    if not os.path.exists(direc):
        os.makedirs(direc)
        
    direc1 = direc+"/"+Gquery[0:len(Gquery)]
    if not os.path.exists(direc1):
        os.makedirs(direc1)
        
    # filename = best.download(quiet=True,filepath=direc1+"/"+Gquery[0:len(Gquery)]+".mp4")
    f = direc1+"/"+Gquery[0:len(Gquery)]+".mp4"
    clip = VideoFileClip(f)
    time = clip.duration
    
    imagenames=[]
    
    for k,c in enumerate(np.arange(0.5,time,1)):
        
        im = clip.get_frame(c)
        imagepath = direc1+"/frames/"
        if os.path.exists(imagepath) and k==0:
            shutil.rmtree(imagepath)
        if not os.path.exists(imagepath):
            os.makedirs(imagepath)
        imagepath = imagepath + str(k)+'.png'
        #scipy.misc.imsave(imagepath, im)
        imageio.imwrite(imagepath, im)
        imagenames.append(imagepath)
    return imagenames