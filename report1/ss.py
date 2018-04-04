##################

# @Author: 			   Chenxin Ma
# @Email: 			   machx9@gmail.com
# @Date:               2018-04-03 13:47:31
# @Last Modified by:   Chenxin Ma
# @Last Modified time: 2018-04-03 13:53:46

##################
import numpy as np
import random
import math
import matplotlib.pyplot as plt


N = 5
# menMeans = (0,0,0.26,0,0)
# womenMeans = (0,0,0.74,0,0)

menMeans = (0,0,0.19,0,0)
womenMeans = (0,0,0.81,0,0)
ind = np.arange(N)    # the x locations for the groups
width = 0.3       # the width of the bars: can also be len(x) sequence


fig = plt.figure(figsize=(6,6))
p1 = plt.bar(ind, menMeans, width)
p2 = plt.bar(ind, womenMeans, width, bottom=menMeans)

plt.ylabel('percentage')
plt.title('D2')
plt.xticks(ind, ('', '', 'CG5', '', ''))
plt.yticks(np.arange(0,1.1,0.1))
plt.legend((p1[0], p2[0]), ('r>1', 'r<1'))

# plt.show()
plt.savefig('ncd2.eps', format='eps')