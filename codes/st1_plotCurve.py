# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:05:46 2018
@author: yann
"""
import matplotlib.pyplot as plt 
import numpy as np
#import time 
#import os 
import re

####################################
filename = '/home/cyyan/projects/CaSoGP/result/log_ValidationFeature.log'
f = open(filename,'r')
content = f.read()
f.close()

epochs = 100
iterations = 149

pattern = re.compile('\*\*\*\*\*\*\*\*_Loss_\*\*\*\*\*\*\*\* \d*.\d*')
loss = [float(one.split(' ')[-1]) for one in pattern.findall(content)]
# count = 0
for epoch in range(epochs,0,-1):
    loss.pop(iterations*epoch-1)

loss2  = np.resize(np.array(loss), (epochs, iterations-1))
loss = np.mean(loss2, axis=1)# (loss2, axis=0)

#plt.figure()
plt.figure(figsize=(10,5))
plt.plot(range(len(loss)), loss, color='red', lw=2)
plt.xlim([0.0, len(loss)])
plt.ylim([0.0, 0.5])
plt.xlabel('Iterations')
plt.ylabel('Loss value')
plt.title('Loss range in iterations for classifier baseline')
# plt.legend(loc="lower right")
plt.savefig('../result/log_ValidationFeature.png')
plt.show()
