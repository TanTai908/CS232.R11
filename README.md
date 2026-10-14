# Secure Steganography – Nén, Mã hóa và Giấu dữ liệu trong ảnh

## 1. Giới thiệu

**Secure Steganography** là dự án ứng dụng kỹ thuật nén dữ liệu, mã hóa và giấu tin trong ảnh nhằm bảo vệ tính bảo mật của tệp tin khi lưu trữ hoặc truyền tải.

Dự án kết hợp ba kỹ thuật chính:

- **Data Compression:** Sử dụng Zlib để nén dữ liệu, giảm kích thước tệp cần giấu.
- **Data Encryption:** Sử dụng AES-256-GCM để mã hóa dữ liệu bằng mật khẩu, đồng thời hỗ trợ kiểm tra tính toàn vẹn của dữ liệu.
- **Image Steganography:** Sử dụng kỹ thuật Least Significant Bit (LSB) để nhúng dữ liệu đã mã hóa vào các bit ít quan trọng của ảnh.

Sau khi nhúng dữ liệu, chương trình tạo ra một ảnh PNG có thể sử dụng như ảnh thông thường. Người nhận cần có ảnh đã nhúng và mật khẩu chính xác để trích xuất, giải mã và khôi phục tệp ban đầu.


## 2. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Xây dựng chương trình xử lý dữ liệu |
| Pillow (PIL) | Đọc, xử lý và lưu ảnh |
| Zlib | Nén và giải nén dữ liệu |
| AES-256-GCM | Mã hóa và xác thực dữ liệu |
| Scrypt | Dẫn xuất khóa mã hóa từ mật khẩu trong phiên bản Python |
| Least Significant Bit (LSB) | Nhúng và trích xuất dữ liệu trong ảnh |
| Gradio | Xây dựng giao diện web bằng Python (nếu sử dụng `app.py`) |
| HTML, CSS, JavaScript | Xây dựng giao diện web chạy trên trình duyệt (nếu sử dụng phiên bản HTML) |

## 3. Nguyên lý hoạt động

### 3.1. Quy trình nhúng dữ liệu

1. **Đọc tệp đầu vào:** Đọc dữ liệu nhị phân của tệp cần giấu, chẳng hạn như DOCX, TXT hoặc PDF.
2. **Nén dữ liệu:** Sử dụng Zlib để giảm kích thước dữ liệu.
3. **Mã hóa dữ liệu:** Dùng mật khẩu để tạo khóa mã hóa và mã hóa dữ liệu bằng AES-256-GCM.
4. **Kiểm tra dung lượng:** Xác định ảnh có đủ sức chứa payload đã mã hóa hay không.
5. **Nhúng dữ liệu:** Ghi dữ liệu vào các bit ít quan trọng của các kênh màu RGB.
6. **Xuất ảnh:** Lưu ảnh kết quả dưới định dạng PNG để hạn chế việc mất dữ liệu do nén ảnh có tổn hao.

### 3.2. Quy trình trích xuất dữ liệu

1. Đọc ảnh đã nhúng dữ liệu.
2. Trích xuất phần dữ liệu được lưu trong các bit LSB.
3. Kiểm tra phần tiêu đề để xác định dữ liệu có đúng định dạng của chương trình hay không.
4. Nhập mật khẩu để giải mã và xác thực dữ liệu.
5. Giải nén dữ liệu bằng Zlib.
6. Khôi phục tệp ban đầu và kiểm tra tính chính xác của dữ liệu.

### 3.3. Sơ đồ tổng quát

**Quá trình nhúng**

Tệp gốc → Nén Zlib → Mã hóa AES-GCM → Nhúng LSB → Ảnh PNG đầu ra

**Quá trình trích xuất**

Ảnh PNG đầu vào → Trích xuất LSB → Giải mã AES-GCM → Giải nén Zlib → Tệp được khôi phục

## 4. Cấu trúc dự án

Cấu trúc tham khảo cho phiên bản Python:

```text
secure_steganography/
├── src/
│   └── steganography.py
├── output/
│   └── stego.png
├── main.py
├── app.py
├── secure_steganography.html
└── README.md
```

- `src/steganography.py`: Cài đặt các hàm nhúng và trích xuất dữ liệu trong ảnh, cùng các hàm mã hóa và giải mã nếu được tổ chức trong mô-đun này.
- `main.py`: Chạy quy trình xử lý chính trên Python.
- `app.py`: Giao diện Gradio, nếu sử dụng phiên bản giao diện Python.
- `secure_steganography.html`: Giao diện HTML/CSS/JavaScript chạy trên trình duyệt.
- `output/`: Lưu ảnh đã nhúng và tệp được khôi phục.

Cấu trúc thực tế có thể khác tùy theo cách tổ chức mã nguồn.

## 5. Cài đặt

### 5.1. Yêu cầu

- Python 3.10 trở lên được khuyến nghị cho môi trường sử dụng các thư viện của dự án.
- pip.
- Ảnh màu RGB làm ảnh chứa dữ liệu.
- Tệp cần giấu.
- Mật khẩu để mã hóa và giải mã.

### 5.2. Cài đặt thư viện

Mở Terminal hoặc PowerShell tại thư mục dự án và chạy:

```bash
pip install pillow cryptography
```

Nếu sử dụng giao diện Gradio:

```bash
pip install gradio
```

Nên sử dụng môi trường ảo để quản lý các thư viện của dự án.

### 6.3. Chạy phiên bản Python

Nếu chương trình chính nằm trong `main.py`:

```bash
python main.py
```

Nếu sử dụng giao diện Gradio:

```bash
python app.py
```

Sau đó mở địa chỉ localhost được hiển thị trong Terminal.

