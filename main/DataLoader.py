from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from main.SamplePreprocessor import preprocess


class Sample:
    "sample from the dataset"

    def __init__(self, coords, id):
        self.coords = coords
        self.id = id


class Batch:
    "batch containing images and ground truth texts"

    def __init__(self, imgs, ids):
        self.imgs = np.stack(imgs, axis=0)
        self.ids = ids



class DataLoader:
    "loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database"

    def __init__(self, text_blocks, batchSize, imgSize, img):
        "loader for dataset at given location, preprocess images and text according to parameters"

        # assert filePath[-1]=='/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.img = img
        self.contour_idx = 0

        for block in text_blocks:
            # put sample into list
            # [x,y,w,h] = block["pts"]
            self.samples.append(Sample(block["pts"],block["id"]))


    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        "iterator"
        return self.currIdx  < len(self.samples)

    def get_snippet(self,coords):
        try:
            [x,y,w,h] = coords
            snippet = self.img[y:y+h,x:x+w]
            gray = cv2.cvtColor(snippet,cv2.COLOR_BGR2GRAY)
            # gray = cv2.threshold(gray,80,255,cv2.THRESH_OTSU)[1]
            cv2.imwrite("contours/"+str(self.contour_idx)+".jpg",gray)
            self.contour_idx+=1
            return gray
        except:
            return ""

    def getNext(self):
        "iterator"
        if self.currIdx+self.batchSize < len(self.samples):
            batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        else:
            batchRange = range(self.currIdx, len(self.samples))
        ids = [self.samples[i].id for i in batchRange]
        imgs = [
            preprocess(self.get_snippet(self.samples[i].coords), self.imgSize, self.dataAugmentation)
            for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(imgs,ids)

# DataLoader("/home/ec2-user/vision/hwrKey.txt",50,(128, 32),32)
