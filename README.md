# Object Detection using SURF

Phát hiện đặc trưng SURF (Speeded-Up Robust Features)

## Mô tả

Dự án này triển khai thuật toán phát hiện đặc trưng **SURF** theo tài liệu OpenCV 3.4.

- **SURF**: Thuật toán nonfree, yêu cầu build OpenCV với `OPENCV_ENABLE_NONFREE=ON` (module `xfeatures2d`).

## Cài đặt

### Yêu cầu

- Python 3.6+
- OpenCV (opencv-python hoặc opencv-contrib-python)

### Cài đặt thư viện

```bash
# Cài đặt phụ thuộc cơ bản
pip install opencv-contrib-python numpy

# OpenCV có SURF cần build thủ công với nonfree (xem bên dưới)
```

**Lưu ý:**

- SURF yêu cầu OpenCV được build với `OPENCV_ENABLE_NONFREE=ON` và module `opencv_contrib/xfeatures2d`.

## Cách sử dụng

### Chạy chương trình

```bash
# Phát hiện đặc trưng SURF trong 1 ảnh
python main.py path/to/image.jpg

# Với hessian threshold tùy chỉnh
python main.py path/to/image.jpg --hessian 300

# Chỉ định thư mục output
python main.py path/to/image.jpg --output-dir my_outputs

# Ví dụ với ảnh trong data
python main.py ./data/imagenette2-320/train/n01440764/ILSVRC2012_val_00000293.JPEG
```

### Quy trình theo yêu cầu (2 bước)

1. **Chuẩn bị dữ liệu**: Đặt ảnh cần phát hiện đặc trưng vào thư mục `data/` hoặc bất kỳ đâu.
2. **Chạy phát hiện SURF**: Truyền đường dẫn ảnh vào chương trình:

```bash
python main.py path/to/image.jpg
```

Kết quả sẽ được lưu trong thư mục `outputs/` với:

- `*_surf_detected.jpg`: Ảnh với keypoints được vẽ
- `*_surf_info.txt`: Thông tin chi tiết về keypoints

### Đầu vào và đầu ra

**Đầu vào:**

- Đường dẫn đến 1 ảnh (`.jpg`, `.jpeg`, `.png`, ...)
- Tùy chọn: `--hessian` để điều chỉnh độ nhạy (mặc định 400)

**Đầu ra:**

- `outputs/*_surf_detected.jpg`: Ảnh với keypoints SURF được đánh dấu (vòng tròn với hướng)
- `outputs/*_surf_info.txt`: Thông tin chi tiết keypoints (tọa độ, size, angle, response)

## Cách hoạt động

### SURF (Speeded-Up Robust Features)

- Yêu cầu OpenCV build với `OPENCV_ENABLE_NONFREE=ON` và `opencv_contrib` (`xfeatures2d`).
- Robust với scale và rotation, descriptor dạng float.
- Tham số chính: Hessian threshold (mặc định 400).

### 2. Quy trình phát hiện đặc trưng

1. **Đọc ảnh:** Đọc ảnh đầu vào và chuyển sang grayscale
2. **Khởi tạo SURF:** Tạo detector SURF với hessian threshold
3. **Phát hiện keypoints:** SURF phát hiện các điểm đặc trưng (corners, blobs)
4. **Tính descriptors:** Tính descriptor 64/128 chiều cho mỗi keypoint
5. **Vẽ và lưu:** Vẽ keypoints lên ảnh gốc và lưu kết quả

### 3. Tham số quan trọng

- **hessian (400):** Ngưỡng Hessian cho SURF. Giá trị cao hơn = ít keypoints hơn nhưng chất lượng cao hơn
- **output_dir (outputs):** Thư mục lưu kết quả

## Cấu trúc code

### Function `detect_surf_features()`

Hàm chính phát hiện đặc trưng SURF:

- Đọc và xử lý ảnh
- Khởi tạo SURF detector
- Phát hiện keypoints và descriptors
- Vẽ và lưu kết quả

### Function `main()`

Hàm CLI để xử lý arguments và gọi detection

## Đặc điểm SURF

**Ưu điểm:**
✅ Bất biến tốt với xoay và tỷ lệ
✅ Robust với thay đổi ánh sáng và góc nhìn
✅ Keypoints có hướng (orientation) rõ ràng
✅ Descriptor 64/128 chiều cho độ phân biệt cao

**Lưu ý:**
⚠️ Yêu cầu OpenCV build với OPENCV_ENABLE_NONFREE
⚠️ Module xfeatures2d từ opencv_contrib

## Mở rộng

Có thể cải thiện chương trình bằng cách:

- Xử lý batch nhiều ảnh cùng lúc
- Thêm feature matching giữa 2 ảnh
- Tích hợp SIFT hoặc ORB để so sánh
- Visualization tốt hơn (heatmap, histogram)
- Export descriptors ra file để sử dụng sau

## Xử lý lỗi SURF

Nếu gặp lỗi:

```
❌ Lỗi: SURF không khả dụng trong OpenCV hiện tại.
Vui lòng build OpenCV với OPENCV_ENABLE_NONFREE=ON và opencv_contrib.
Chi tiết: module 'cv2.xfeatures2d' has no attribute 'SURF_create'
```

**Nguyên nhân:**
OpenCV cài qua pip (`opencv-python` hoặc `opencv-contrib-python`) không bao gồm các thuật toán nonfree như SURF do hạn chế bản quyền.

**Giải pháp: Build OpenCV từ source với NONFREE enabled**

### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
# Chạy script build tự động
chmod +x build_opencv_with_surf.sh
./build_opencv_with_surf.sh
```

Script sẽ:

- Cài đặt dependencies (cmake, pkg-config, eigen, tbb, ...)
- Clone opencv và opencv_contrib từ GitHub
- Cấu hình CMake với `OPENCV_ENABLE_NONFREE=ON`
- Build và cài đặt OpenCV (mất ~30-60 phút)

### Cách 2: Build thủ công

#### Bước 1: Cài đặt dependencies

```bash
# Cài Homebrew nếu chưa có: https://brew.sh
brew install cmake pkg-config wget
brew install jpeg libpng libtiff openexr
brew install eigen tbb
```

#### Bước 2: Download source code

```bash
mkdir ~/opencv_build && cd ~/opencv_build

# Clone OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x
cd ..

# Clone opencv_contrib (chứa xfeatures2d/SURF)
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.x
cd ..
```

#### Bước 3: Cấu hình CMake

```bash
mkdir -p opencv/build && cd opencv/build

# Lấy Python path
PYTHON_EXECUTABLE=$(which python3)
PYTHON_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

# Chạy CMake với NONFREE enabled
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D PYTHON3_EXECUTABLE=$PYTHON_EXECUTABLE \
      -D PYTHON3_PACKAGES_PATH=$PYTHON_PACKAGES \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D WITH_TBB=ON \
      -D WITH_EIGEN=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF ..
```

**Các tham số quan trọng:**

- `OPENCV_ENABLE_NONFREE=ON`: Kích hoạt SURF, SIFT và các thuật toán nonfree
- `OPENCV_EXTRA_MODULES_PATH`: Đường dẫn đến opencv_contrib/modules
- `PYTHON3_EXECUTABLE`: Đảm bảo build cho đúng Python version
- `WITH_TBB=ON`: Tăng tốc xử lý đa luồng

#### Bước 4: Build và cài đặt

```bash
# Build (sử dụng tất cả CPU cores)
make -j$(sysctl -n hw.ncpu)

# Cài đặt (cần quyền sudo)
sudo make install
```

**Lưu ý:** Quá trình build có thể mất 30-60 phút tùy cấu hình máy.

#### Bước 5: Kiểm tra cài đặt

```bash
# Kiểm tra version và SURF
python3 -c "import cv2; print('OpenCV version:', cv2.__version__); print('SURF available:', hasattr(cv2.xfeatures2d, 'SURF_create'))"

# Test SURF
python3 -c "import cv2; surf = cv2.xfeatures2d.SURF_create(); print('SURF initialized successfully!')"
```

Nếu thành công, bạn sẽ thấy:

```
OpenCV version: 4.x.x
SURF available: True
SURF initialized successfully!
```

#### Bước 6: Dọn dẹp (tùy chọn)

```bash
# Xóa thư mục build để tiết kiệm dung lượng
rm -rf ~/opencv_build
```

### Xử lý sự cố

**Lỗi: CMake không tìm thấy Python**

```bash
# Chỉ định rõ Python path
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")
```

**Lỗi: Build failed do thiếu dependencies**

```bash
# Cài đầy đủ dependencies
brew install cmake pkg-config jpeg libpng libtiff openexr eigen tbb hdf5
```

**OpenCV bị conflict với version cũ**

```bash
# Gỡ OpenCV cũ
pip uninstall opencv-python opencv-contrib-python -y

# Xóa build cũ và build lại
rm -rf ~/opencv_build/opencv/build
mkdir ~/opencv_build/opencv/build
# ... tiếp tục từ bước CMake
```

## Tham khảo

- Bay, H., Tuytelaars, T., & Van Gool, L. (2006). SURF: Speeded up robust features.
- Rublee, E., et al. (2011). ORB: An efficient alternative to SIFT or SURF.
- OpenCV Documentation: https://docs.opencv.org/
