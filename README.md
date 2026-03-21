# Sonar Returns Classification – Final Project

**Môn:** Trí Tuệ Nhân Tạo  
**Bài toán:** Binary Classification – Mine (M) vs Rock (R)  
**Dataset:** Sonar Returns (sonar.csv) – 208 mẫu, 60 features

---

## Nhật ký phát triển

### Giai đoạn 1 – Khảo sát và thiết kế

Xuất phát điểm là đề bài yêu cầu xây dựng pipeline phân loại nhị phân trên bộ dữ liệu sonar. Sau khi đọc kỹ đề, nhóm xác định cần thực hiện đủ 7 bước: định nghĩa vấn đề, phân tích dữ liệu, chuẩn bị dữ liệu, xây dựng model, đánh giá bằng cross-validation, kiểm nghiệm trên test set và tinh chỉnh tham số.

Thay vì đặt toàn bộ code vào một notebook duy nhất, nhóm quyết định tách thành 3 notebook riêng biệt theo chức năng: EDA, Train và Test. Quyết định này xuất phát từ nhu cầu thực tế: muốn train trên một bộ dữ liệu nhưng test trên bộ khác (cross-test) mà không phải chạy lại toàn bộ từ đầu.

Cấu trúc thư mục được thiết kế như sau:

```
final-project-CSTTNT/
├── bin/          ← script tự động hóa pipeline
├── data/         ← file CSV dữ liệu
├── doc/          ← báo cáo, slide
├── exps_/        ← kết quả thực nghiệm (CSV, Excel, biểu đồ)
├── model/        ← model .pkl đã train
└── prj/          ← notebook chính
```

---

### Giai đoạn 2 – Xây dựng EDA (01_eda.ipynb)

EDA được xây dựng với tham số `DATA_NAME` ở đầu notebook để dễ dàng chuyển đổi giữa các bộ dữ liệu mà không cần sửa code bên dưới. Mọi đường dẫn file output đều dùng `f'{DATA_NAME}_...'` làm prefix để tránh ghi đè khi chạy nhiều bộ dữ liệu.

Các bước trong EDA:

1. Khai báo thư viện và tham số
2. Đọc dữ liệu
3. Thống kê mô tả (shape, dtypes, count/mean/min/max/percentile)
4. Kiểm tra missing values và duplicates
5. Phân phối lớp – bar chart và pie chart
6. Histogram phân phối feature theo lớp
7. Boxplot phát hiện outlier
8. Violin plot
9. Giá trị trung bình feature theo lớp
10. Correlation heatmap
11. Kiểm tra phân phối chuẩn (Shapiro-Wilk)
12. Chia train/test 70/30 (stratified, seed=42)
13. Chuẩn hóa: Raw, MinMax, Standard – lưu scaler `.pkl`
14. So sánh phân phối trước/sau chuẩn hóa
15. Ghi kết quả vào `{DATA_NAME}_eda_log.xlsx`

Một vấn đề phát sinh khi chạy trên Windows: file Excel bị lỗi `PermissionError` nếu đang mở trong Excel. Giải pháp là dùng `mode='w'` kết hợp đọc lại lịch sử cũ trước khi ghi đè.

---

### Giai đoạn 3 – Xây dựng Train (02_train.ipynb)

Notebook train được thiết kế với nguyên tắc **chỉ train, không test**. Kết quả test được tách hoàn toàn sang `03_test.ipynb` để đảm bảo tính độc lập giữa hai bước.

Trước khi train, notebook tự động kiểm tra EDA đã chạy chưa bằng cách xác minh sự tồn tại của `{TRAIN_DATA_NAME}_eda_log.xlsx` và 6 file CSV tương ứng. Nếu thiếu, notebook từ chối train và hướng dẫn chạy EDA trước.

Pipeline train gồm các bước:

- **Bước 5:** Khởi tạo thí nghiệm – khai báo `TRAIN_DATA_NAME`, đường dẫn, tham số
- **Bước 6:** Kiểm tra điều kiện EDA
- **Bước 7:** Baseline – chạy 10 model với tham số mặc định qua 5-Fold Stratified CV, vẽ biểu đồ so sánh
- **Bước 7.2:** Tinh chỉnh SVM (GridSearch: C × kernel = 12 tổ hợp × 5 folds) và Random Forest (9 tổ hợp × 5 folds)
- **Bước 8:** Fit toàn bộ tập train, lưu tất cả model `.pkl` vào `model/{TRAIN_DATA_NAME}/`
- **Bước 9:** Ghi `{TRAIN_DATA_NAME}_train_log.xlsx`

10 model được dùng: kNN, Naive Bayes, SVM, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, LDA, MLP, Logistic Regression.

Một lỗi quan trọng được phát hiện và sửa trong quá trình phát triển: hàm AUC ban đầu lấy cứng cột `[:,1]` từ `predict_proba`, dẫn đến AUC ra giá trị sai (0.06–0.12 thay vì 0.5–1.0). Nguyên nhân là không biết cột 1 tương ứng với lớp nào. Sửa bằng cách tìm đúng cột qua `list(model.classes_).index(pos_label)`.

---

### Giai đoạn 4 – Xây dựng Test (03_test.ipynb)

Notebook test nhận 2 tham số độc lập:

- `TRAIN_DATA_NAME` – tên bộ đã train (để load model)
- `TEST_DATA_NAME` – tên bộ muốn test (có thể khác train)

Thiết kế này cho phép **cross-test**: dùng model train trên sonar để test trên ionosphere, hoặc ngược lại, mà không cần train lại.

Trước khi test, notebook kiểm tra 3 điều kiện:
1. Folder `model/{TRAIN_DATA_NAME}/` tồn tại và có file `.pkl`
2. File `{TEST_DATA_NAME}_eda_log.xlsx` tồn tại (bộ test đã EDA)
3. Cả 3 file CSV test (`raw`, `minmax`, `standard`) tồn tại

Kết quả test bao gồm: bảng Accuracy/Precision/Recall/F1/AUC (2 chữ số thập phân), phân tích top 2 model theo từng tiêu chí, confusion matrix của best model, ROC curves và ghi vào `{TRAIN}__test_{TEST}_log.xlsx`.

---

### Giai đoạn 5 – Tự động hóa với bin scripts

4 script trong `bin/` được xây dựng để chạy pipeline từ terminal mà không cần mở Jupyter:

```bash
python bin/run_eda.py sonar
python bin/run_train.py sonar
python bin/run_test.py sonar sonar
python bin/run_all.py sonar
```

Mỗi script có bước kiểm tra điều kiện riêng trước khi chạy – nếu thiếu điều kiện sẽ báo lỗi rõ ràng và hướng dẫn cách fix. Ban đầu dùng `papermill` để chạy notebook và tạo file output mới mỗi lần. Sau đó chuyển sang `jupyter nbconvert --inplace` để chạy thẳng trên file gốc, không tạo thêm file thừa trong `prj/`.

---

### Giai đoạn 6 – Sửa lỗi và cải thiện

Trong quá trình phát triển, một số vấn đề được phát hiện và xử lý:

**Tên file output thiếu prefix:** Ban đầu EDA xuất ra `eda_log.xlsx`, `train_raw.csv`... Khi chạy nhiều bộ dữ liệu, các file bị ghi đè lên nhau. Giải pháp là thêm `{DATA_NAME}_` làm prefix cho tất cả file output.

**AUC tính sai:** Đã mô tả ở Giai đoạn 3.

**Kết quả 4 chữ số thập phân:** Đề yêu cầu giữ 2 chữ số. Sửa toàn bộ `round(..., 4)` và `:.4f` thành 2 chữ số trong cả 3 notebook.

**Bước kiểm tra EDA trong train:** Ban đầu không có – nếu chạy train khi chưa EDA sẽ lỗi `FileNotFoundError` khó hiểu. Thêm bước kiểm tra tường minh với thông báo rõ ràng.

**Biểu đồ bị lệch giá trị:** Lỗi do dùng `row._5` (truy cập cột theo vị trí số) trong `itertuples()` thay vì tên cột. Sửa bằng cách dùng `subset['Mean Acc']` trực tiếp.

---

### Giai đoạn 7 – Hoàn thiện

Các việc hoàn thiện cuối:

- Thêm phân tích **Top 2 model theo từng tiêu chí** vào `03_test.ipynb` theo yêu cầu đề
- Tách phần test ra khỏi `02_train.ipynb` để đúng thiết kế
- Cập nhật cấu trúc thư mục thêm `doc/` và `test/` cho đồng bộ với project mẫu
- Viết README này

---

## Cách chạy

```bash
# Cài đặt thư viện
pip install scikit-learn pandas numpy matplotlib seaborn openpyxl jupyter papermill

# Tải dữ liệu
curl -o data/sonar.csv https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv

# Chạy toàn bộ pipeline
python bin/run_all.py sonar

# Hoặc chạy từng bước
python bin/run_eda.py sonar
python bin/run_train.py sonar
python bin/run_test.py sonar sonar
```

---

## Kết quả

- Best model: **SVM (tuned)** với scaler Standard
- Accuracy: ~0.90 trên tập test
- File kết quả: `exps_/sonar_train_log.xlsx`, `exps_/sonar__test_sonar_log.xlsx`
- Model: `model/sonar/sonar__best_model.pkl`