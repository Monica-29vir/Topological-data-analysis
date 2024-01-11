import os
from matplotlib import pyplot as plt
from skimage import io, color
import numpy as np
from ripser import Rips
from persim import PersistenceImager
from skimage.transform import resize

# 加载图像并转换为灰度
image_path = '/media/yanghan/T7/1-5Test/Train/000006.jpg'
img = io.imread(image_path)
img_gray = color.rgb2gray(img)

# H0 的拓扑数据分析
rips_h0 = Rips()
dgms_h0 = rips_h0.fit_transform(img_gray.T)
H0_dgm = dgms_h0[0]

# 用最大有限值替换无穷大
max_finite_value = np.max(H0_dgm[H0_dgm[:, 1] != np.inf, 1])
H0_dgm[H0_dgm[:, 1] == np.inf, 1] = max_finite_value

# 添加一个新的点到 H0_dgm
new_point_h0 = np.array([[0.5, 0.6]])
H0_dgm = np.vstack([H0_dgm, new_point_h0])

# 创建并绘制 H0 持久图
pimgr_h0 = PersistenceImager(pixel_size=0.1)
pimgr_h0.fit(H0_dgm)
persistence_image_h0 = pimgr_h0.transform(H0_dgm)
pimgr_h0.kernel_params = {'sigma': 0.5}
plt.imshow(persistence_image_h0, cmap='viridis')
plt.title('persistence_image_h0')
plt.colorbar()
plt.show()
# H1 的拓扑数据分析
rips_h1 = Rips()
dgms_h1 = rips_h1.fit_transform(img_gray.T)
H1_dgm = dgms_h1[1]

# 创建并绘制 H1 持久图
pimgr_h1 = PersistenceImager(pixel_size=0.1)
pimgr_h1.fit(H1_dgm)
persistence_image_h1 = pimgr_h1.transform(H1_dgm)
pimgr_h1.kernel_params = {'sigma': 0.5}
# 可以绘制融合图像
plt.imshow(persistence_image_h1, cmap='viridis')
plt.title('persistence_image_h1')
plt.colorbar()
plt.show()

# 确定统一的数组大小并进行零填充
max_shape = (max(persistence_image_h0.shape[0], persistence_image_h1.shape[0]), max(persistence_image_h0.shape[1], persistence_image_h1.shape[1]))

padded_H0_dgm = np.zeros(max_shape)
padded_H1_dgm = np.zeros(max_shape)

padded_H0_dgm[:persistence_image_h0.shape[0], :persistence_image_h0.shape[1]] = persistence_image_h0
padded_H1_dgm[:persistence_image_h1.shape[0], :persistence_image_h1.shape[1]] = persistence_image_h1

normalized_persistence_image_h0_padded = (padded_H0_dgm - np.min(padded_H0_dgm)) / (np.max(padded_H0_dgm) - np.min(padded_H0_dgm))
normalized_persistence_image_h1_padded = (padded_H1_dgm - np.min(padded_H1_dgm)) / (np.max(padded_H1_dgm) - np.min(padded_H1_dgm))

# 使用最大值进行堆叠
fused_image = np.maximum(normalized_persistence_image_h0_padded, normalized_persistence_image_h1_padded)

# 将图像大小调整为224x224
Final_PI = resize(fused_image, (224, 224), anti_aliasing=True)
Final_PI_H0_224 = resize(normalized_persistence_image_h0_padded, (224, 224), anti_aliasing=True)
Final_PI_H1_224 = resize(normalized_persistence_image_h1_padded, (224, 224), anti_aliasing=True)

# 将浮点图像转换为8位整数格式
Final_PI_uint8 = (Final_PI * 255).astype(np.uint8)
Final_PI_H0_224_uint8 = (Final_PI_H0_224 * 255).astype(np.uint8)
Final_PI_H1_224_uint8 = (Final_PI_H1_224 * 255).astype(np.uint8)

# 将三个图像堆叠为一个三通道的图像
Final_combined = np.stack([Final_PI_uint8, Final_PI_H0_224_uint8, Final_PI_H1_224_uint8], axis=-1)

# 获取原始图像文件名（不包含扩展名）
image_filename = os.path.splitext(os.path.basename(image_path))[0]

# 构造保存路径，保持一致的文件名
output_path = os.path.join('/media/yanghan/T7/test-PI', f'{image_filename}_Final_combined.jpg')

# 保存处理后的图片
io.imsave(output_path, Final_combined, quality=95)
# 可以绘制融合图像
plt.imshow(Final_combined)
plt.title('Final_combined')
plt.colorbar()
plt.show()
