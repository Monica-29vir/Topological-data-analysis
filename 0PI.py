from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt
from ripser import Rips
from persim import PersistenceImager

image_path = '/home/yanghan/桌面/荷花.jpg'
img = io.imread(image_path)
img_gray = color.rgb2gray(img)

rips = Rips(maxdim=1, coeff=2)
dgms = rips.fit_transform(img_gray.T)
H0_dgm = dgms[0]
H1_dgm = dgms[1]

# 将0维特征的无穷大的值替换为最大有限值
max_finite_value = np.max(H0_dgm[H0_dgm[:, 1] != np.inf, 1])
H0_dgm[H0_dgm[:, 1] == np.inf, 1] = max_finite_value

new_point=np.array([[3,4]])
H0_dgm= np.vstack([H0_dgm,new_point])

plt.figure(figsize=(10,5))
plt.subplot()
rips.plot(dgms, legend=False, show=False)
plt.title("Persistence diagram")
plt.show()

pimgr = PersistenceImager(pixel_size=0.1)  # 持久图像的分辨率是通过选择像素大小来调整的
pimgr.fit(H0_dgm)
fig, axs = plt.subplots(1, 2, figsize=(20,5))
ax = axs[1]
pimgr.plot_diagram(H0_dgm, skew=True, ax=axs[0])

axs[0].set_title('H0 birth-persistence diagram', fontsize=16)
persistence_image = pimgr.transform(H0_dgm)
pimgr.kernel_params = {'sigma': 0.5} # 对于默认的二元正态高斯核，控制扩散sigma的参数可以通过浮点或2x2协方差矩阵指定
pimgr.plot_image(pimgr.transform(H0_dgm), ax=axs[1])
axs[1].set_title('persistence image', fontsize=16)
plt.tight_layout()
plt.show()
