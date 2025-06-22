# Dự báo tiêu thụ điện năng (Nhóm 1 - Lớp 64TTNT2)

## Giới thiệu

Dự án nhằm xây dựng và đánh giá các mô hình dự báo tiêu thụ điện năng dựa trên dữ liệu tiêu thụ điện gia đình. Chúng tôi sử dụng hai kiến trúc chính:

- **LSTM (Long Short-Term Memory)**: mạng nơ-ron hồi quy dài-ngắn hạn.
- **Transformer**: mô hình cơ chế tự chú ý (self-attention).
- **GRU**
- **TimeNet**

Dữ liệu gốc được thu thập từ tập dữ liệu [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption), đã làm sạch và chuẩn hóa.

## Nguồn dataset
- [Data thô](https://drive.google.com/drive/folders/18WCjjPg5-utygi88xtnSJt_oYm--GmzD?usp=sharing)
- [Data clean](https://drive.google.com/file/d/1pyba1vkSOhfM4lXCJ7sH0DHp0IXy4DMM/view?usp=drive_link)

## Cấu trúc thư mục

```
├── images/                   # Hình ảnh, biểu đồ minh họa
├── models/                   # Mã nguồn định nghĩa mô hình
├── utils/                    # Các hàm hỗ trợ, xử lý dữ liệu chung
├── weights/                  # File trọng số (weights) đã huấn luyện sẵn
├── prepare_data.ipynb        # Notebook tiền xử lý và chuẩn bị dữ liệu
├── train_LSTM.ipynb          # Notebook huấn luyện mô hình LSTM
├── train_Transformer.ipynb   # Notebook huấn luyện mô hình Transformer
├── predict_LSTM.ipynb        # Notebook dự báo với mô hình LSTM
├── predict_Transformer.ipynb # Notebook dự báo với mô hình Transformer
├── requirements.txt          # Danh sách thư viện cần cài đặt
└── README.md                 # Tài liệu hướng dẫn này
```

## Cài đặt

1. **Clone** repository:

   ```bash
   git clone https://github.com/nthnguynn/Nhom1_64TTNT2_DuBaoDienNang.git
   cd Nhom1_64TTNT2_DuBaoDienNang
   ```

2. **Tạo môi trường ảo** (khuyến nghị):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\\Scripts\\activate  # Windows
   ```

3. **Cài đặt** các phụ thuộc:

   ```bash
   pip install -r requirements.txt
   ```

## Hướng dẫn sử dụng

1. **Chuẩn bị dữ liệu**:

   - Mở `prepare_data.ipynb` và thực thi các cell để đọc, làm sạch, phân tách dữ liệu thành tập huấn luyện và kiểm thử.

2. **Huấn luyện mô hình**:

   - **LSTM**: Mở và chạy `train_LSTM.ipynb` để huấn luyện, theo dõi biểu đồ mất mát và lưu trọng số.
   - **Transformer**: Mở và chạy `train_Transformer.ipynb` tương tự.

3. **Dự báo và đánh giá**:

   - **LSTM**: Mở `predict_LSTM.ipynb`, load mô hình và chạy dự báo trên tập kiểm thử.
   - **Transformer**: Mở `predict_Transformer.ipynb` để thực thi.

4. **Kết quả**:

   - Kết quả dự báo, biểu đồ so sánh và các chỉ số đánh giá (MSE, MAE) được lưu trong thư mục `images/`.
   - Trọng số và mô hình học được lưu trong `weights/`.
  
## Cam kết

Nhóm cam kết đây là sản phẩm do nhóm tự thực hiện. Quá trình làm việc có tham khảo một số tài liệu và mã nguồn open–source để học hỏi, nhưng tất cả ý tưởng, thiết kế và mã nguồn đã được nhóm nghiên cứu, hiểu rõ và tự tay triển khai. 

**Cảm ơn thầy cô và các bạn đã ghé thăm! Chúc mọi người có một ngày tốt lành.**
