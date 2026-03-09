# SURF Feature Matching

So khớp đặc trưng SURF (Speeded-Up Robust Features) giữa 2 ảnh

📖 **[Quick Start Guide](QUICKSTART.md)** | 📋 **[OpenCV Setup với Pyenv](OPENCV_PYENV_SETUP.md)**

## Mô tả

Dự án này triển khai thuật toán **so khớp đặc trưng SURF** giữa hai ảnh theo tài liệu OpenCV 3.4.

- **SURF**: Thuật toán nonfree, yêu cầu build OpenCV với `OPENCV_ENABLE_NONFREE=ON` (module `xfeatures2d`)
- Chương trình phát hiện đặc trưng SURF trong hai ảnh và tìm các điểm tương ứng giữa chúng

## Cài đặt

### Yêu cầu

- Python 3.6+
- OpenCV với SURF support (yêu cầu build từ source với `OPENCV_ENABLE_NONFREE=ON`)

### Cài đặt thư viện

```bash
# Cài đặt phụ thuộc cơ bản
pip install numpy
```

**Lưu ý quan trọng:**

- SURF yêu cầu OpenCV được build với `OPENCV_ENABLE_NONFREE=ON` và module `opencv_contrib/xfeatures2d`
- OpenCV từ pip (`opencv-python`, `opencv-contrib-python`) **KHÔNG** có SURF
- Bạn cần build OpenCV từ source

### Build OpenCV với SURF Support

#### Cách 1: Tự động với Makefile (Khuyến nghị)

```bash
# Chạy một lệnh để build toàn bộ (macOS/Linux/WSL)
make all
```

Makefile sẽ tự động:
- Kiểm tra pyenv setup
- Cài đặt dependencies (cmake, pkg-config, v.v.)
- Download OpenCV và opencv_contrib source
- Configure CMake với `OPENCV_ENABLE_NONFREE=ON`
- Build và install vào pyenv environment
- Verify cài đặt

**Các lệnh hữu ích:**
```bash
make help           # Xem trợ giúp
make check-pyenv    # Kiểm tra pyenv
make verify         # Kiểm tra sau khi build
make clean          # Xóa build directory
```

#### Cách 2: Build thủ công

Xem hướng dẫn chi tiết trong [OPENCV_PYENV_SETUP.md](OPENCV_PYENV_SETUP.md) để build thủ công từng bước.

**Tóm tắt:**
1. Cài dependencies: `cmake`, `pkg-config`, etc.
2. Download OpenCV và opencv_contrib từ GitHub
3. Configure với CMake (set `OPENCV_ENABLE_NONFREE=ON`)
4. Build: `make -j$(nproc)`
5. Install: `make install`

### Kiểm tra cài đặt

```bash
# Kiểm tra OpenCV version
python3 -c "import cv2; print(cv2.__version__)"

# Kiểm tra SURF có hoạt động không
python3 -c "import cv2; surf = cv2.xfeatures2d.SURF_create(); print('✓ SURF works!')"

# Hoặc dùng Makefile
make verify
```

## Cách sử dụng

### Chạy chương trình

```bash
# So khớp đặc trưng SURF giữa 2 ảnh
python main.py path/to/image1.jpg path/to/image2.jpg

# Chỉ định thư mục output
python main.py path/to/image1.jpg path/to/image2.jpg --output-dir my_outputs

# Ví dụ với ảnh trong data
python main.py ./data/messi_1.jpg ./data/messi_2.jpg
```

### Quy trình hoạt động

1. **Chuẩn bị dữ liệu**: Đặt 2 ảnh cần so khớp vào thư mục `data/` hoặc bất kỳ đâu
2. **Chạy so khớp SURF**: Truyền đường dẫn 2 ảnh vào chương trình:

```bash
python main.py image1.jpg image2.jpg
```

Kết quả sẽ được lưu trong thư mục `outputs/` với:

- `image1_surf_info.txt`: Thông tin keypoints của ảnh 1
- `image2_surf_info.txt`: Thông tin keypoints của ảnh 2
- `image1_vs_image2_surf_match.jpg`: Ảnh kết hợp (composite) bao gồm:
  - Panel trên: Ảnh 1 với keypoints được vẽ
  - Panel giữa: Ảnh 2 với keypoints được vẽ
  - Panel dưới: Đường nối giữa các keypoints so khớp

### Đầu vào và đầu ra

**Đầu vào:**

- Đường dẫn đến ảnh gốc/tham chiếu (image1)
- Đường dẫn đến ảnh cần so khớp (image2)
- Tùy chọn: `--output-dir` để chỉ định thư mục lưu kết quả (mặc định: `outputs`)

**Đầu ra:**

- `image1_surf_info.txt`: Thông tin chi tiết keypoints của ảnh 1 (tọa độ, size, angle, response)
- `image2_surf_info.txt`: Thông tin chi tiết keypoints của ảnh 2
- `image1_vs_image2_surf_match.jpg`: Ảnh kết hợp 3 panel:
  - Panel 1: Ảnh 1 với keypoints SURF được đánh dấu
  - Panel 2: Ảnh 2 với keypoints SURF được đánh dấu
  - Panel 3: Đường nối giữa các keypoints so khớp (tối đa 100 matches tốt nhất)

## Cách hoạt động

### 1. SURF (Speeded-Up Robust Features)

- Yêu cầu OpenCV build với `OPENCV_ENABLE_NONFREE=ON` và `opencv_contrib` (`xfeatures2d`)
- Robust với scale và rotation, descriptor dạng float
- Tham số chính: Hessian threshold (cố định = 400 trong code)

### 2. Quy trình so khớp đặc trưng

1. **Đọc 2 ảnh:** Đọc cả hai ảnh đầu vào và chuyển sang grayscale
2. **Khởi tạo SURF:** Tạo detector SURF với hessian threshold = 400
3. **Phát hiện keypoints:** SURF phát hiện các điểm đặc trưng trong cả 2 ảnh
4. **Tính descriptors:** Tính descriptor cho mỗi keypoint của cả 2 ảnh
5. **So khớp đặc trưng:** Sử dụng BFMatcher (Brute Force Matcher) với NORM_L2 và crossCheck=True
6. **Sắp xếp matches:** Sắp xếp theo distance và lấy tối đa 100 matches tốt nhất
7. **Vẽ và lưu:**
   - Vẽ keypoints lên mỗi ảnh
   - Vẽ đường nối giữa các keypoints so khớp
   - Ghép 3 panel thành 1 ảnh composite theo chiều dọc

### 3. Các hàm chính

- **detect_surf_features(image_path, output_dir):** Phát hiện keypoints SURF trong 1 ảnh
- **match_two_images(image1_path, image2_path, output_dir):** So khớp SURF giữa 2 ảnh
- **parse_args():** Xử lý command-line arguments
- **main():** Hàm chính điều khiển luồng chương trình

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
