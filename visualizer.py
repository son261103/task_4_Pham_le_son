# visualizer.py
import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self):
        self.metrics = {
            'accuracy': 'Độ chính xác',
            'precision': 'Độ chuẩn xác',
            'recall': 'Độ bao phủ',
            'f1_score': 'Điểm F1'
        }
        self.times = {
            'training_time': 'Thời gian huấn luyện',
            'prediction_time': 'Thời gian dự đoán'
        }

    def plot_results(self, results):
        """
        Vẽ biểu đồ kết quả
        """
        splits = list(results.keys())

        # Vẽ biểu đồ metrics
        plt.figure(figsize=(15, 10))
        for idx, (metric, metric_name) in enumerate(self.metrics.items(), 1):
            plt.subplot(2, 2, idx)
            svm_values = [results[split]['svm'][metric] for split in splits]
            knn_values = [results[split]['knn'][metric] for split in splits]

            plt.plot(splits, svm_values, 'o-', label='SVM')
            plt.plot(splits, knn_values, 's-', label='KNN')
            plt.title(f'{metric_name} theo tỷ lệ Train-Test')
            plt.xlabel('Tỷ lệ Train-Test')
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Vẽ biểu đồ thời gian
        plt.figure(figsize=(12, 6))
        for idx, (timing, time_name) in enumerate(self.times.items(), 1):
            plt.subplot(1, 2, idx)
            svm_times = [results[split]['svm'][timing] for split in splits]
            knn_times = [results[split]['knn'][timing] for split in splits]

            plt.plot(splits, svm_times, 'o-', label='SVM')
            plt.plot(splits, knn_times, 's-', label='KNN')
            plt.title(f'{time_name} theo tỷ lệ Train-Test')
            plt.xlabel('Tỷ lệ Train-Test')
            plt.ylabel('Thời gian (giây)')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def print_detailed_results(self, results):
        """
        In kết quả chi tiết
        """
        print("\nKẾT QUẢ CHI TIẾT")
        print("=" * 80)

        for split, split_results in results.items():
            print(f"\nTỷ lệ Train-Test: {split}")
            print("-" * 80)

            for model in ['svm', 'knn']:
                print(f"\nKết quả mô hình {model.upper()}:")
                print(f"Thời gian huấn luyện: {split_results[model]['training_time']:.2f} giây")
                print(f"Thời gian dự đoán: {split_results[model]['prediction_time']:.2f} giây")
                print(f"Độ chính xác (Accuracy): {split_results[model]['accuracy']:.4f}")
                print(f"Độ chuẩn xác (Precision): {split_results[model]['precision']:.4f}")
                print(f"Độ bao phủ (Recall): {split_results[model]['recall']:.4f}")
                print(f"Điểm F1 (F1-Score): {split_results[model]['f1_score']:.4f}")