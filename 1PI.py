from ripser import Rips
from persim import PersistenceImager
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

image_path = '/media/yanghan/T7/1-5Test/Train/000006.jpg'
img = io.imread(image_path)
img_gray = color.rgb2gray(img)

#拓扑数据分析
rips = Rips()
dgms = rips.fit_transform(img_gray.T)
H0_dgm = dgms[0]
H1_dgm = dgms[1]

plt.figure(figsize=(10,5))
plt.subplot()
rips.plot(dgms, legend=False, show=False)
plt.title("Persistence diagram")
plt.show()
pimgr = PersistenceImager(pixel_size=0.1)  # 持久图像的分辨率是通过选择像素大小来调整的
pimgr.fit(H1_dgm)
fig, axs = plt.subplots(1, 2, figsize=(20,5))
pimgr.plot_diagram(H1_dgm, skew=True, ax=axs[0])
axs[0].set_title('H1 birth-persistence diagram', fontsize=16)
pimgr.kernel_params = {'sigma': 0.5} # 对于默认的二元正态高斯核，控制扩散sigma的参数可以通过浮点或2x2协方差矩阵指定
pimgr.plot_image(pimgr.transform(H1_dgm), ax=axs[1])
axs[1].set_title('persistence image', fontsize=16)
plt.tight_layout()
plt.show()