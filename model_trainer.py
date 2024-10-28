# model_trainer.py
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelTrainer:
    def __init__(self):
        self.results = {}

    def train_and_evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Huấn luyện và đánh giá một mô hình
        """
        # Huấn luyện
        train_start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - train_start

        # Dự đoán
        pred_start = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - pred_start

        # Tính toán các metrics
        metrics = {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'predictions': y_pred
        }

        return metrics

    def run_experiment(self, X, y, test_sizes=[0.2, 0.3, 0.4, 0.6]):
        """
        Chạy thử nghiệm với các tỷ lệ train-test khác nhau
        """
        for test_size in test_sizes:
            print(f"\nĐang thực hiện thử nghiệm với tỷ lệ test = {test_size}")
            train_size = 1 - test_size

            # Chia dữ liệu
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Khởi tạo mô hình
            svm_model = SVC(kernel='rbf', random_state=42)
            knn_model = KNeighborsClassifier(n_neighbors=5)

            # Huấn luyện và đánh giá
            svm_results = self.train_and_evaluate_model(
                svm_model, X_train, X_test, y_train, y_test, 'SVM'
            )
            knn_results = self.train_and_evaluate_model(
                knn_model, X_train, X_test, y_train, y_test, 'KNN'
            )

            # Lưu kết quả
            self.results[f"{int(train_size * 100)}-{int(test_size * 100)}"] = {
                'svm': svm_results,
                'knn': knn_results
            }

        return self.results