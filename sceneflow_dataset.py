# encoding: utf-8

import os
import sys
import re
import time
from scipy.misc import imread
import numpy as np
import skimage.io
import matplotlib.pyplot as plt


class Dataset(object):

    def __init__(self, SceneFlowDir, isTraining=True):

        self.leftImgPath=[]
        self.leftGtPath=[]
        self.rightImgPath=[]
        self.rightGtPath=[]
        self.sceneSize = []
        self.sceneIndexDict={}
        self.sceneSizeNumber = 0
        self.SceneFlowDir = SceneFlowDir
        self.isTraining = isTraining

    def readDrivingPathFromDirection(self, mm, forOrBack, slowOrfast):

        drivingImgPath = 'frames_cleanpass/' + mm + '/' + forOrBack + '/' + slowOrfast + '/' + 'left/'
        cDrivingImgPath = 'frames_cleanpass/' + mm + '/' + forOrBack + '/' + slowOrfast + '/' + 'right/'
        gtDrivingImgPath = 'disparity/' + mm + '/' + forOrBack + '/' + slowOrfast + '/' + 'left/'
        cGtDrivingImgPath = 'disparity/' + mm + '/' + forOrBack + '/' + slowOrfast + '/' + 'right/'

        # drivingImgPath = 'frames_cleanpass/15mm_focallength/scene_forwards/slow/left/'
        # cDrivingImgPath = 'frames_cleanpass/15mm_focallength/scene_forwards/slow/right/'
        # gtDrivingImgPath = 'disparity/15mm_focallength/scene_forwards/slow/left/'
        # cGtDrivingImgPath = 'disparity/15mm_focallength/scene_forwards/slow/right/'

        self.leftImgPath.append(os.path.join(
            os.path.join(self.SceneFlowDir, 'driving'), drivingImgPath)
        )

        self.rightImgPath.append(os.path.join(
            os.path.join(self.SceneFlowDir, 'driving'), cDrivingImgPath)
        )

        self.leftGtPath.append(os.path.join(
            os.path.join(self.SceneFlowDir, 'driving'), gtDrivingImgPath)

        )

        self.rightGtPath.append(os.path.join(
            os.path.join(self.SceneFlowDir, 'driving'), cGtDrivingImgPath)
        )


    def readPathFromDirection(self):

        if self.isTraining:

            if not os.path.isdir(self.SceneFlowDir):
                raise ValueError('Not found: %s\r\n' % (self.SceneFlowDir))
            sceneModule = os.listdir(self.SceneFlowDir)

            for moduleName in sceneModule:

                if moduleName == 'driving':

                    self.readDrivingPathFromDirection('15mm_focallength','scene_forwards','slow')
                    self.readDrivingPathFromDirection('15mm_focallength','scene_forwards','fast')
                    self.readDrivingPathFromDirection('15mm_focallength','scene_backwards','slow')
                    self.readDrivingPathFromDirection('15mm_focallength','scene_backwards','fast')

                    self.readDrivingPathFromDirection('35mm_focallength', 'scene_forwards', 'slow')
                    self.readDrivingPathFromDirection('35mm_focallength', 'scene_forwards', 'fast')
                    self.readDrivingPathFromDirection('35mm_focallength', 'scene_backwards', 'slow')
                    self.readDrivingPathFromDirection('35mm_focallength', 'scene_backwards', 'fast')

                elif moduleName == 'flyingthings3d':

                    OneDir = os.path.join(self.SceneFlowDir, moduleName) # flying3d
                    OneSubDirList = os.listdir(OneDir)
                    for oneSubDir in OneSubDirList:
                        if oneSubDir == 'frames_cleanpass':
                            TwoDir = os.path.join(os.path.join(OneDir, oneSubDir), 'TRAIN') # flying3d/Train
                            TwoDirList = os.listdir(TwoDir)
                            for TwoName in TwoDirList:
                                self.leftImgPath, self.rightImgPath = self.readSceneFromZeroDir(
                                    self.leftImgPath, self.rightImgPath, TwoDir, TwoName)
                        if oneSubDir == 'disparity':
                            TwoDir = os.path.join(os.path.join(OneDir, oneSubDir), 'TRAIN')  # flying3d/Train
                            TwoDirList = os.listdir(TwoDir)
                            for TwoName in TwoDirList:
                                self.leftGtPath, self.rightGtPath = self.readSceneFromZeroDir(
                                    self.leftGtPath, self.rightGtPath, TwoDir, TwoName)



                elif moduleName == 'monkaa':
                    monkaaDir = os.path.join(self.SceneFlowDir, moduleName) # monkaa
                    monkaaList = os.listdir(monkaaDir)
                    for monkaa in monkaaList:
                        if monkaa == 'disparity':
                            self.leftGtPath, self.rightGtPath = self.readSceneFromZeroDir(
                                self.leftGtPath, self.rightGtPath, monkaaDir, monkaa)

                        elif monkaa == 'frames_cleanpass':
                            self.leftImgPath, self.rightImgPath = self.readSceneFromZeroDir(
                                self.leftImgPath, self.rightImgPath, monkaaDir, monkaa)


        else:

            OneDir = os.path.join(self.SceneFlowDir, 'flyingthings3d')  # flyingthings3d
            OneSubDirList = os.listdir(OneDir)
            for oneSubDir in OneSubDirList:
                if oneSubDir == 'frames_cleanpass':
                    TwoDir = os.path.join(os.path.join(OneDir, oneSubDir), 'TEST')  # flying3d/TEST
                    TwoDirList = os.listdir(TwoDir)
                    for TwoName in TwoDirList:
                        self.leftImgPath, self.rightImgPath = self.readSceneFromZeroDir(
                            self.leftImgPath, self.rightImgPath, TwoDir, TwoName)
                if oneSubDir == 'disparity':
                    TwoDir = os.path.join(os.path.join(OneDir, oneSubDir), 'TEST')  # flying3d/TEST
                    TwoDirList = os.listdir(TwoDir)
                    for TwoName in TwoDirList:
                        self.leftGtPath, self.rightGtPath = self.readSceneFromZeroDir(
                            self.leftGtPath, self.rightGtPath, TwoDir, TwoName)

        return self.leftImgPath, self.rightImgPath, self.leftGtPath, self.rightGtPath

    def readSceneFromZeroDir(self, firstList, secondList, zeroDir, first):

        _firstList = firstList
        _secondList = secondList
        firstDir = os.path.join(zeroDir, first)
        firstList = os.listdir(firstDir)
        for thing in firstList:
            thingDir = os.path.join(firstDir, thing)
            _firstList.append(os.path.join(thingDir, 'left'))
            _secondList.append(os.path.join(thingDir, 'right'))

        return _firstList, _secondList


    def getSceneSizeAndSceneSizeNumber(self):
        for leftImgPathName in self.leftImgPath:
            self.sceneSize.append(len(os.listdir(leftImgPathName)))


        for leftImgPathSize in self.sceneSize:
            self.sceneSizeNumber  = self.sceneSizeNumber + leftImgPathSize

        return self.sceneSize, self.sceneSizeNumber

    def getBatch(self):

        tmpIndexCounter = 0
        indexList = []
        sceneIndexList = []

        # index scene module

        for leftImgPathSize in self.sceneSize:
            for sceneSizeNumber in range(leftImgPathSize):
                sceneIndexList.append(tmpIndexCounter)
            tmpIndexCounter = tmpIndexCounter + 1


            if leftImgPathSize == 800 or leftImgPathSize == 300:
                tmpIndexNumber = 0
                for number in range(leftImgPathSize):
                    # tmpIndexNumber = tmpIndexNumber + 1
                    indexList.append(tmpIndexNumber)
                    tmpIndexNumber = tmpIndexNumber + 1



            elif leftImgPathSize == 10:
                tmpIndexNumber = 6
                for number in range(leftImgPathSize):
                    indexList.append(tmpIndexNumber)
                    tmpIndexNumber = tmpIndexNumber + 1

            else:
                tmpIndexNumber = 0
                for number in range(leftImgPathSize):
                    indexList.append(tmpIndexNumber)
                    tmpIndexNumber = tmpIndexNumber + 1

        return indexList, sceneIndexList

    def getRandomIndex(self):
        randomIndex = np.arange(self.sceneSizeNumber)
        np.random.shuffle(randomIndex)

        return randomIndex

class GetBatchFromDataSet(object):

    def __init__(self, moduleIndexList, itemIndexList, randomGetBatchIndex, imgList, cImgList, gtList, cGtList, sceneSizeList,
                 batchSize=1, isGray=True, hOri=540, wOri=960, hTarget=256, wTarget=512):
        self.hOri = hOri
        self.wOri = wOri
        self.hTarget = hTarget
        self.wTarget = wTarget
        self.batchSize = batchSize
        self.isGray = isGray
        self.moduleIndexList = moduleIndexList
        self.itemIndexList = itemIndexList
        self.randomGetBatchIndex = randomGetBatchIndex
        self.imgList = imgList
        self.cImgList = cImgList
        self.gtList = gtList
        self.cGtList = cGtList
        self.sceneSizeList = sceneSizeList

    def GetWholeDir(self, randomIndex):

        moduleIndex = self.moduleIndexList[randomIndex]
        itemIndex = self.itemIndexList[randomIndex]
        imgDir, cImgDir = self.GetImgDirFromIndex(moduleIndex, itemIndex, self.imgList, self.cImgList)
        gtDir, cGtDir = self.GetDptDirFromIndex(moduleIndex, itemIndex, self.gtList, self.cGtList)

        return imgDir, cImgDir, gtDir, cGtDir


    def GetImgDirFromIndex(self, moduleIndex, itemIndex, dirList, cDirList):

        moduleDir = dirList[moduleIndex]
        cModuleDir = cDirList[moduleIndex]
        size = self.sceneSizeList[moduleIndex]
        itemDir = self.getItemNumber(size, itemIndex) + '.png'

        ImgDir = os.path.join(moduleDir, itemDir)
        cImgDir = os.path.join(cModuleDir, itemDir)

        if not os.path.isfile(ImgDir):
            print('Warning: not found: %s, and ignore\r\n'%(ImgDir))
        if not os.path.isfile(cImgDir):
            print('Warning: not found: %s, and ignore\r\n'%(cImgDir))


        return ImgDir, cImgDir

    def GetDptDirFromIndex(self, moduleIndex, itemIndex, dirList, cDirList):

        moduleDir = dirList[moduleIndex]
        cModuleDir = cDirList[moduleIndex]
        size = self.sceneSizeList[moduleIndex]
        itemDir = self.getItemNumber(size, itemIndex) + '.pfm'
        gtDir = os.path.join(moduleDir, itemDir)
        cGtDir = os.path.join(cModuleDir, itemDir)

        if not os.path.isfile(gtDir):
            print('Warning: not found: %s, and ignore\r\n'%(gtDir))
        if not os.path.isfile(cGtDir):
            print('Warning: not found: %s, and ignore\r\n'%(cGtDir))

        # print(flag)


        return gtDir, cGtDir

    def getItemNumber(self, size, itemNumber):
        if size == 800 or size == 300:
            head = ''
            if itemNumber == 0:
                head = '0001'
            if itemNumber and itemNumber <= 8:
                head = '000' + str(itemNumber + 1)  ### 0001~0009
            if itemNumber >= 9 and itemNumber <= 98:
                head = '00' + str(itemNumber + 1)  ###  0010~0099
            if itemNumber >= 99:
                head = '0' + str(itemNumber + 1)  ###   0100~0800

        elif size == 10:
            head = ''
            if itemNumber<10:
                head = '000' + str(itemNumber)
            else:
                head = '00' + str(itemNumber)

        else:
            head = ''
            if itemNumber == 0:
                head = '0000'
            if itemNumber and itemNumber <= 9:
                head = '000' + str(itemNumber)
            if itemNumber >= 10 and itemNumber <= 99:
                head = '00' + str(itemNumber)
            if itemNumber >= 100:
                head = '0' + str(itemNumber)

        return head


    def GetDataFromDir(self, imgDir, cImgDir, gtDir, cGtDir):

        img = self.GetImgFromDir(imgDir)
        cImg = self.GetImgFromDir(cImgDir)
        gt = self.GetDptFromDir(gtDir)
        cGt = self.GetDptFromDir(cGtDir)

        return img, cImg, gt, cGt

    def GetImgFromDir(self, dir):

        # suffix = os.path.splitext(dir)
        # if suffix == '.png':
        img = imread(dir, mode='L')
        imgnd = np.array(img, dtype='float32')
        imgnd1 = imgnd / 128.0
        imgnd2 = imgnd1 - 1.0
        # else:
        #     print('Warning: not valid image direction: %s, and ignore\r\n' % (dir))
        #     imgnd2 = np.zeros((self.hOri, self.wOri))

        return imgnd2

    def GetDptFromDir(self, dir):

        # suffix = os.path.splitext(dir)
        # if suffix == '.pfm':

        dpt = open(dir)
        dpt_data, _= self.load(dpt)
        # else:
        #     print('Warning: not valid disparity direction: %s, and ignore\r\n' % (dir))
        #     dpt_data = np.zeros((self.hOri, self.wOri))

        return dpt_data

    def load(self, file):
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        datanew = np.reshape(data, shape)
        datanew1 = datanew[::-1, :]
        return datanew1, scale

    def crop(self, trainimg_left, trainimg_right, traindpt_left, traindpt_right, H_in, W_in, H_out, W_out, batchsize):

            H_ori = H_in
            W_ori = W_in
            H = H_out
            W = W_out
            H_delta = H_ori - H
            W_delta = W_ori - W
            batch_size = batchsize
            train_img_left = trainimg_left
            train_img_right = trainimg_right
            train_dpt_left = traindpt_left
            train_dpt_right = traindpt_right

            images_left = np.zeros((batch_size, H, W), dtype=np.float32)
            images_right = np.zeros((batch_size, H, W), dtype=np.float32)
            disparity_left = np.zeros((batch_size, H, W))
            disparity_right = np.zeros((batch_size, H, W))
            loc_y = (np.random.random_sample((batch_size, 1)) * H_delta).astype(int)
            loc_x = (np.random.random_sample((batch_size, 1)) * W_delta).astype(int)
            loc = np.append(loc_y, loc_x, axis=1)

            for p in range(batch_size):
                images_left[p, :, :] = train_img_left[loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]
                images_right[p, :, :] = train_img_right[loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]
                disparity_left[p, :, :] = train_dpt_left[loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]
                disparity_right[p, :, :] = train_dpt_right[loc[p, 0]:loc[p, 0] + H, loc[p, 1]:loc[p, 1] + W]

            return images_left, images_right, disparity_left, disparity_right


def getIndexLists(isTraining):

    rootDir = '/media/home_bak/share/Dataset/SceneFlow-dataset/'
    sfDataSet = Dataset(rootDir, isTraining=isTraining)
    leftImgPath, rightImgPath, leftGtPath, rightGtPath = sfDataSet.readPathFromDirection()
    sceneDirSizeNumberList, number1 = sfDataSet.getSceneSizeAndSceneSizeNumber()
    indexList, sceneIndexList= sfDataSet.getBatch()
    randomIndexArray = sfDataSet.getRandomIndex()

    return leftImgPath, rightImgPath, leftGtPath, rightGtPath, sceneDirSizeNumberList, indexList, sceneIndexList, randomIndexArray

def getBatchData(sceneIndexList, indexList, randomIndexArray,result1, result2, result3, result4, sceneDirSizeNumberList, iterNumber):

    dataGenerator = GetBatchFromDataSet(sceneIndexList, indexList, randomIndexArray, result1, result2, result3, result4, sceneDirSizeNumberList)
    iter = randomIndexArray[iterNumber]
    imgDir, cImgDir, gtDir, cGtDir = dataGenerator.GetWholeDir(iter)
    img, cImg, gt, cGt = dataGenerator.GetDataFromDir(imgDir, cImgDir, gtDir, cGtDir)
    cropImg, cropCImg, cropGt, cropCGt = dataGenerator.crop(img, cImg, gt, cGt, 540, 960, 256, 512, 1)

    return cropImg, cropCImg, cropGt, cropCGt

def textdata(ndarray):

    return np.max(ndarray), np.min(ndarray), np.mean(ndarray)

if __name__ == '__main__':
    result1, result2, result3, result4, sceneDirSizeNumberList, indexList, sceneIndexList, randomIndexArray = getIndexLists(True)
    test1, test2, test3, test4, testSceneDirSizeNumberList, testIndexList, testSceneIndexList, testRandomIndexArray = getIndexLists(False)
    print(len(indexList), len(sceneIndexList), randomIndexArray.shape)
    print(len(testIndexList), len(testSceneIndexList), testRandomIndexArray.shape)

    # for i in range(len(result1)):
    #     cropImg, cropCImg, cropGt, cropCGt = getBatchData(sceneIndexList, indexList, randomIndexArray, result1, result2, result3, result4,
    #                 sceneDirSizeNumberList, i)
    #     print(cropImg.shape, cropCImg.shape, cropGt.shape, cropCGt.shape)
    #     print(textdata(cropImg))
    #     print(textdata(cropCImg))
    #     print(textdata(cropGt))
    #     print(textdata(cropCGt))





