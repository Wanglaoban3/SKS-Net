import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def calculate_aspect_ratio(widths, heights):
    aspect_ratios = np.divide(widths, heights)
    return aspect_ratios.reshape(-1, 1)

def cluster_images(widths, heights, n_clusters):
    aspect_ratios = calculate_aspect_ratio(widths, heights)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(aspect_ratios)
    centers = kmeans.cluster_centers_
    return labels, centers

def display_clusters(widths, heights, labels):
    plt.scatter(widths, heights, c=labels)
    plt.xlabel('宽度')
    plt.ylabel('高度')
    plt.title('图片聚类结果')
    plt.show()

def find_min_max_ratios(labels, aspect_ratios):
    min_ratios = np.zeros(n_clusters)
    max_ratios = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_ratios = aspect_ratios[cluster_indices]
        min_ratios[i] = np.min(cluster_ratios)
        max_ratios[i] = np.max(cluster_ratios)
    return min_ratios, max_ratios

image_widths = []  # 图片的宽度列表
image_heights = []  # 图片的高度列表
root_path = 'C:/wrd/Magnetic'
for path in os.listdir(root_path):
    for sub_path in os.listdir(os.path.join(root_path, path)):
        for sub_sub_path in os.listdir(os.path.join(root_path, path, sub_path)):
            if '.jpg' in sub_sub_path:
                img = Image.open(os.path.join(root_path, path, sub_path, sub_sub_path))
                image_widths.append(img.width)
                image_heights.append(img.height)


n_clusters = 6  # 聚类数量

image_labels, cluster_centers = cluster_images(image_widths, image_heights, n_clusters)

display_clusters(image_widths, image_heights, image_labels)

aspect_ratios = calculate_aspect_ratio(image_widths, image_heights)

min_ratios, max_ratios = find_min_max_ratios(image_labels, aspect_ratios)

for i in range(n_clusters):
    print(f"聚类 {i}：最小长宽比 = {min_ratios[i]}, 最大长宽比 = {max_ratios[i]}, 中心 = {cluster_centers[i]}")