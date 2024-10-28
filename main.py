# main.py
import time
from data_loader import DataLoader
from model_trainer import ModelTrainer
from visualizer import Visualizer


def main():
    # Cấu hình
    base_path = r"D:\Lean_for_future\Thi_giac_may_tinh\Panoramic radiographs with periapical lesions Dataset\Panoramic radiographs with periapical lesions Dataset\Periapical Dataset\Periapical Lesions"
    num_images = 100

    print("BẮT ĐẦU THỰC HIỆN THỬ NGHIỆM PHÂN LỚP ẢNH Y TẾ")
    print("=" * 80)

    # Khởi tạo các module
    data_loader = DataLoader(base_path, num_images)
    model_trainer = ModelTrainer()
    visualizer = Visualizer()

    # Tải và tiền xử lý dữ liệu
    start_time = time.time()
    X, y = data_loader.load_and_preprocess_images()
    X_scaled = data_loader.scale_features(X)
    preprocessing_time = time.time() - start_time
    print(f"Tiền xử lý hoàn thành trong {preprocessing_time:.2f} giây")

    # Huấn luyện và đánh giá mô hình
    results = model_trainer.run_experiment(X_scaled, y)

    # Hiển thị kết quả
    visualizer.print_detailed_results(results)
    visualizer.plot_results(results)


if __name__ == "__main__":
    main()