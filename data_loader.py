# data_loader.py
import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, base_path, num_images=100):
        self.base_path = base_path
        self.num_images = num_images

    def load_and_preprocess_images(self):
        """
        Tải và tiền xử lý ảnh từ đường dẫn đã cho
        """
        print("Đang tải và tiền xử lý ảnh...")
        images = []
        labels = []
        classes = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]

        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(self.base_path, class_name)
            files = os.listdir(class_path)[:self.num_images]

            for file_name in files:
                img_path = os.path.join(class_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    img_flat = img.flatten()
                    images.append(img_flat)
                    labels.append(class_idx)

        return np.array(images), np.array(labels)

    def scale_features(self, X):
        """
        Chuẩn hóa đặc trưng
        """
        scaler = StandardScaler()
        return scaler.fit_transform(X)
