# Mô tả bộ dữ liệu VNLP (Vietnamese License Plate)

Bộ dữ liệu VNLP được sử dụng cho huấn luyện và kiểm thử mô hình nhận diện biển số xe cơ giới tại Việt Nam.

## 1. Cấu trúc và phân loại
Bộ dữ liệu gồm 3 thư mục tương ứng với 3 dạng biển số thực tế:
- `one_row/`: Biển số ô tô dạng 1 dòng (dài). Tọa độ và nhãn lưu trong `list_one_row.txt`.
- `two_rows/`: Biển số ô tô dạng 2 dòng (vuông). Tọa độ và nhãn lưu trong `list_two_row.txt`.
- `two_rows_label_xe_may/`: Biển số xe máy 2 dòng. Tọa độ và nhãn lưu trong `list_two_rows_label_xe_may.txt`.

## 2. Quy ước định dạng tên file ảnh
Tên file ảnh trong dataset được mã hóa chứa thông tin nhãn và tọa độ bounding box biển số:
`[loại]_[id]_[layout]_[biển_số]_[x1]_[y1]_[x2]_[y2].jpg`

Ví dụ:
`1_1010_1_36A38587_102_344_227_413.jpg`
- `1_1010`: ID ảnh
- `1`: Định dạng biển số (1: 1 dòng, 2: 2 dòng)
- `36A38587`: Nhãn chuỗi ký tự biển số xe (Thanh Hóa, seri A, số 385.87)
- `[102, 344, 227, 413]`: Tọa độ bounding box (x_min, y_min, x_max, y_max) trong ảnh gốc.

## 3. Danh mục nhãn và file danh sách
- `list_one_row.txt`: 19.086 mẫu biển 1 dòng
- `list_two_row.txt`: 12.618 mẫu biển ô tô 2 dòng
- `list_two_rows_label_xe_may.txt`: 5.593 mẫu biển xe máy 2 dòng
- `VNLP_readme`: Ghi chú kỹ thuật của bộ dữ liệu gốc.
