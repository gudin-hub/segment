import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataFile = r'Subject_01.mat' # 单个的mat文件
data = scio.loadmat(dataFile)
print(type(data))
# print (data['data'])
# 由于导入的mat文件是structure类型的，所以需要取出需要的数据矩阵
a=data['images']
# 取出需要的数据矩阵

# 数据矩阵转图片的函数
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

new_im = MatrixToImage(a)
plt.imshow(a, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
new_im.save('Subject_01.bmp') # 保存图片
