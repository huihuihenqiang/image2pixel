import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def pixelate_and_cluster_colors(input_image_path, output_image_path, size=(32, 32), eps=10, min_samples=5):
    # 打开图片
    img = Image.open(input_image_path)

    # 将图片缩小到指定尺寸，像素化
    img = img.resize(size, Image.BICUBIC)

    # 转换为RGB
    img = img.convert("RGB")

    # 将图片转换为numpy数组
    img_array = np.array(img)

    # 获取图像的所有像素值，并将其扁平化
    pixels = img_array.reshape(-1, 3)

    # 使用DBSCAN进行颜色聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(pixels)

    # 对于每个聚类标签，计算它的代表颜色（即聚类中心）
    unique_labels = np.unique(dbscan_labels)
    centers = []
    for label in unique_labels:
        if label != -1:  # 排除噪声标签
            cluster_pixels = pixels[dbscan_labels == label]
            cluster_center = np.mean(cluster_pixels, axis=0).astype(int)
            centers.append(cluster_center)

    # 将每个像素替换为其对应的聚类中心颜色
    new_img_array = np.copy(pixels)
    for i, label in enumerate(dbscan_labels):
        if label != -1:  # 排除噪声标签
            new_img_array[i] = centers[label]

    # 重新构建图像
    new_img_array = new_img_array.reshape(img_array.shape)

    # 将聚类后的图像保存为新的图片
    new_img = Image.fromarray(new_img_array.astype('uint8'))
    new_img.save(output_image_path)

    # 显示结果
    plt.imshow(new_img)
    plt.axis('off')
    plt.show()


# 使用示例
input_image_path = 'input2.jpg'  # 输入图片路径
output_image_path = 'output_pixelated.png'  # 输出图片路径
pixelate_and_cluster_colors(input_image_path, output_image_path, size=(40, 40), eps=5, min_samples=4)