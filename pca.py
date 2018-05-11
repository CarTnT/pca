# -*- coding: utf-8 -*-
from numpy import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import operator
import cv2
#定义一个全局特征转换变量　这个变量是在PCA中求出的
global redEigVects
def shortestDis(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1) #axis=0, 表示列。axis=1, 表示行。
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #classCount={}
    '''
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    '''
    return sortedDistIndicies[0]
def pca(dataMat, topNfeat=9999999):
    global redEigVects
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)#sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]   #reorganize eig vects largest to smallest
    #得到低维度数据
    print(redEigVects.shape, dataMat.shape)
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
# 读取trainingMat
filename = '/Users/mac/pca/multimedia/training/%d/%d.jpg'
trainingNumbers = 1
imageWidth = 180
imageHeight = 200
#初始化trainingMat
size = (int(imageWidth*0.2), int(imageHeight*0.2))
size1 = (int(imageWidth*0.6), int(imageHeight*0.6))
trainingMat0=zeros((19,36*40))
for i in range(1,20):
    for j in range(trainingNumbers):
        im = cv2.imread(filename %(i,(j+1)),0)
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        im = np.array(im)
        im = im.flatten()
        #trainingMat0[(i-1)*5 + j]=im
        trainingMat0[i-1]=im
#PCA
DD = 49
#trainingMat,reconMat=pca(trainingMat0,DD)
#eigendigits
'''
fig = plt.figure()
for i in range(0,9):
    im = trainingMat0[i]
    im = np.array(im)
    im = im.reshape(40,36)
    im = np.real(im)
    plotwindow = fig.add_subplot('33%d'%((i+1)%9))
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
#plt.title('D = %d' % DD)
plt.show()

# 读取test图像
'''
fig = plt.figure()
tfilename = "/Users/mac/pca/multimedia/testing/%d.jpg"
testNumber = 19
count = 0
errCount = 0
errRate = []
y = np.arange(1,100,5)
for m in y:
    errCount = 0
    trainingMat,reconMat=pca(trainingMat0,m)
    for i in range(testNumber):
        im2 = cv2.imread(tfilename %(i+1),0)
        im2 = cv2.resize(im2, size, interpolation=cv2.INTER_AREA)
        im2 = np.array(im2)
        im2 = im2.flatten()
        meanVals = mean(im2, axis=0)
        meanRemoved = im2 - meanVals #remove mean
        #这个时候使用的降维特征变量为上边给训练数组得出的特征量
        testingMat=meanRemoved*redEigVects
        index = shortestDis(testingMat.getA(), trainingMat.getA())
        if index != i:
            errCount += 1
            print(index,i)
    print(errCount/testNumber)
    errRate.append(errCount/testNumber)
plt.plot(y, errRate,label = 'k',linewidth = 2,color = 'red')
plt.axis([0,50,0,1])
for i in range(1,len(y)):
    plt.text(y[i],errRate[i],str(float('%.2f' % errRate[i])), family='serif', style='italic', ha='right')
plt.grid(True)
plt.title('errRate change with D')
plt.show()
'''
    #显示匹配人脸
    count += 1
    im4 = cv2.imread(tfilename %(i+1))
    im4 = cv2.resize(im4, size1, interpolation=cv2.INTER_AREA)
    #ax = fig.add_subplot(testNumber,2,count)
    ax = fig.add_subplot(testNumber,2,count)
    ax.set_title("Testing image %d matches training image %d" % (i+1, index+1))
    plt.imshow(im4)
    plt.xticks([])
    plt.yticks([])
    count += 1
    im3 = cv2.imread(filename %((index+1)/5 + 1, index%5 + 1))
    print(filename %((index+1)/5 + 1, index%5 + 1),im3.shape)
    im3 = cv2.resize(im3, size1, interpolation=cv2.INTER_AREA)
    ax = fig.add_subplot(testNumber,2,count)
    ax.set_title("Testing image %d matches training image %d" % (i+1, index+1))
    plt.imshow(im3)
    plt.xticks([])
    plt.yticks([])
    #print("Testing image %d matches training image %d" % (i+1, index+1) )
plt.axis("off")#去除坐标轴
plt.show()
'''
