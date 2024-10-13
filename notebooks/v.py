import os 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import visdom

vis = visdom.Visdom()

path = './videos/bedroom'
win = vis.image(np.random.rand(3, 1024, 1024), opts={'title': 'image1', 'caption': 'How random.'})

for i in os.listdir(path)[:10]:
    img = cv2.imread(path + '/' + i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    vis.image(img, opts={'title':i}, win=win)