# SURF Object Tracking

Ứng dụng tracking object trong video sử dụng **SURF** (Speeded-Up Robust Features) kết hợp **Template Matching** (NCC), giao diện PyQt5.

## Yêu cầu hệ thống

- Windows 10/11 (64-bit)
- Dung lượng trống: ~3 GB (cho Miniconda + packages)

## Cài đặt chi tiết

### Bước 1: Cài Miniconda

1. Tải Miniconda cho Windows 64-bit:
   ```
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
   ```

2. Chạy file `.exe`, cài đặt vào thư mục mặc định hoặc tùy chọn (ví dụ `D:\Miniconda3`).

3. Trong quá trình cài:
   - Chọn **"Just Me"**
   - **Không** cần tick "Add to PATH" (sẽ dùng đường dẫn trực tiếp)

4. Kiểm tra cài thành công — mở **Anaconda Prompt** (tìm trong Start Menu) và gõ:
   ```bash
   conda --version
   ```

### Bước 2: Tạo môi trường conda

Mở **Anaconda Prompt** và chạy lần lượt:

```bash
conda create -n surf python=3.7 -y
conda activate surf
```

> **Lưu ý:** Bắt buộc dùng Python 3.7 vì `opencv-contrib-python==3.4.2.16` không hỗ trợ Python >= 3.8.

### Bước 3: Cài các thư viện

Trong môi trường `surf` đã activate:

```bash
pip install opencv-contrib-python==3.4.2.16
pip install PyQt5==5.15.10
pip install numpy==1.21.6
```

Kiểm tra SURF hoạt động:

```bash
python -c "import cv2; s = cv2.xfeatures2d.SURF_create(); print('SURF OK, OpenCV', cv2.__version__)"
```

Kết quả mong đợi:
```
SURF OK, OpenCV 3.4.2
```

### Bước 4: Chạy ứng dụng

#### Cách 1: Dùng Anaconda Prompt

```bash
conda activate surf
cd đường\dẫn\tới\surf-tracking
python object_tracking_surf.py
```

#### Cách 2: Dùng file run.bat (Windows)

1. Mở file `run.bat` bằng Notepad, sửa đường dẫn Miniconda cho đúng:
   ```bat
   @echo off
   D:\Miniconda3\envs\surf\python.exe "object_tracking_surf.py"
   pause
   ```
   > Thay `D:\Miniconda3` bằng thư mục bạn đã cài Miniconda ở Bước 1.

2. Double-click `run.bat` để chạy.

#### Cách 3: Dùng Git Bash

```bash
/d/Miniconda3/envs/surf/python.exe "object_tracking_surf.py"
```

## Hướng dẫn sử dụng

1. **Mở video**: Bấm **Browse** → chọn file `.mp4` (hoặc `.avi`, `.mov`, `.mkv`)
2. **Phát video**: Bấm **Start**
3. **Chọn đối tượng**: Bấm **"Chọn ROI"** → video tự pause → kéo chuột trên video để chọn vùng object cần track
4. **Tracking tự động**: Sau khi chọn ROI, video tự resume và hiển thị:
   - Khung xanh bao quanh object
   - `objX`, `objY`: tọa độ tâm (chuẩn hóa -1 đến 1)
   - `area`: diện tích object (% so với frame)
   - `FPS`: tốc độ xử lý
5. **Khi mất tracking**: Tự động bật lại chế độ chọn ROI
6. **Pause/Resume**: Bấm **Pause**
7. **Reset**: Bấm **Reset** để xóa tracker, chọn lại ROI

## Cấu trúc thư mục

```
surf-tracking/
├── object_tracking_surf.py   # Code chính (tracker + GUI)
├── test_tracker.py           # Unit tests
├── run.bat                   # Launcher Windows
├── run.sh                    # Launcher Bash
├── plan.md                   # Tài liệu thiết kế
├── README.md                 # File này
└── videos/                   # Video mẫu
    ├── human.mp4
    └── vehicle_highway.mp4
```

## Chạy tests

```bash
conda activate surf
python test_tracker.py
```

Kết quả mong đợi: `36 passed, 0 failed — ALL TESTS PASSED!`

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `ModuleNotFoundError: No module named 'cv2'` | Chưa cài OpenCV | `pip install opencv-contrib-python==3.4.2.16` |
| `module 'cv2' has no attribute 'xfeatures2d'` | Cài nhầm `opencv-python` (không có contrib) | `pip uninstall opencv-python && pip install opencv-contrib-python==3.4.2.16` |
| `SURF_create() error` | Python >= 3.8 hoặc OpenCV >= 4.0 | Tạo lại env với Python 3.7 |
| `No module named 'PyQt5'` | Chưa cài PyQt5 | `pip install PyQt5==5.15.10` |
| Video không mở được | Codec thiếu hoặc file hỏng | Thử video khác, đảm bảo file `.mp4` hợp lệ |
| Tracking lost ngay lập tức | Object quá nhỏ hoặc không có texture | Chọn ROI lớn hơn, object cần có pattern/texture rõ ràng |

## Thông tin kỹ thuật

- **Tracker**: SURF feature matching + Template Matching (NCC) hybrid
- **SURF**: Hessian threshold = 100, extended 128-dim descriptors
- **Matcher**: BFMatcher (L2 norm) + Lowe's ratio test
- **Fallback**: Multi-scale NCC template matching (5 scales: 0.9–1.1)
- **Adaptive update**: Blend template + SURF features mỗi 5 frames thành công
- **Validation**: Homography check, aspect ratio check, area ratio check
