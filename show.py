import math
from random import randint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_image_from_array(csvImages):

    ln = len(csvImages)

    fig, axes = plt.subplots(int(ln**0.5+0.99999), int(ln**0.5+0.99999))
    axes = [i for j in axes for i in j]

    for i in range(ln):
        row = csvImages[i]

        if len(row) == 784:
            label = row[0]
            image = [row[28*i:28*(i+1)] for i in range(28)]
            title1 = f'Index — Label: {label}'
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(title1)

    for i in range(ln, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()





def load_images_from_csv(csv_path='test.csv', mx = 10000):
    df = pd.read_csv(csv_path)
    rows = df.iloc[list(range(min(mx, len(df))))].values.astype(np.uint8)
    labels = rows[:, 0]
    images = [list(r) for r in rows[:, 1:]]
    return labels, images



def getRandomRots(images, mxRot = 30):

    tempSc = 56
    half = tempSc/2
    result = []
    for i in range(len(images)):
        rot = randint(-mxRot, mxRot)*math.pi/180
        res = [0 for i in range(tempSc**2)]
        for x in range(tempSc):
            for y in range(tempSc):
                xx = (x-half)*math.cos(rot)+(y-half)*math.sin(rot)+half
                yy = (y-half)*math.cos(rot)-(x-half)*math.sin(rot)+half
                xx,yy = int(xx*28/tempSc),int(yy*28/tempSc)
                if (xx<0 or yy<0) or (xx>=28 or yy>=28):
                    continue
                res[y*tempSc+x] = int(images[i][yy*28+xx])
        res1 = [0 for i in range(28*28)]
        for x in range(tempSc):
            for y in range(tempSc):
                xx, yy = int(x*28/tempSc), int(y*28/tempSc)
                res1[yy*28+xx] += res[y*tempSc+x]

        result.append([int(i/tempSc*28) for i in res1])

    return result

def getRandomOffset(images, mxOff=5):
    result = []

    for i in range(len(images)):
        mvx, mvy = randint(-mxOff, mxOff),randint(-mxOff, mxOff)
        res = [0 for i in range(28*28)]

        for x in range(28):
            for y in range(28):
                xx, yy = x+mvx, y+mvy
                if (xx<0 or yy<0) or (xx>=28 or yy>=28):
                    continue
                res[yy*28+xx] = images[i][y*28+x]
        result.append(res)

    return result

def getRandomScale(images, mnSc = 0.75, mxScale = 1):

    tempSc = 56
    half = tempSc/2
    result = []

    for i in range(len(images)):
        scale = randint(int(mnSc*100), int(mxScale*100))/100
        res = [0 for i in range(tempSc**2)]

        for x in range(tempSc):
            for y in range(tempSc):
                xx, yy = int((x-half)/scale)+half, int((y-half)/scale)+half
                xx, yy = int(xx*28/tempSc),int(yy*28/tempSc)
                if (xx<0 or yy<0) or (xx>=28 or yy>=28):
                    continue
                res[y*tempSc+x] = int(images[i][yy*28+xx])
        res1 = [0 for i in range(28*28)]
        for x in range(tempSc):
            for y in range(tempSc):
                xx, yy = int(x*28/tempSc), int(y*28/tempSc)
                res1[yy*28+xx] += res[y*tempSc+x]

        result.append([int(i/tempSc*28) for i in res1])
    return result


