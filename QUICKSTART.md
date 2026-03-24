# Quick Start Guide

Hướng dẫn nhanh để setup và chạy SURF Feature Matching

## 🚀 Setup Nhanh (macOS/Linux/WSL)

### Bước 1: Build OpenCV với SURF

```bash
# Một lệnh để setup toàn bộ (30-60 phút)
make all
```

Lệnh này sẽ:
- ✓ Kiểm tra pyenv environment
- ✓ Cài đặt dependencies (cmake, pkg-config, v.v.)
- ✓ Download OpenCV và opencv_contrib
- ✓ Configure với OPENCV_ENABLE_NONFREE=ON
- ✓ Build OpenCV
- ✓ Install vào pyenv environment
- ✓ Verify cài đặt

### Bước 2: Kiểm tra

```bash
# Kiểm tra SURF có hoạt động
python3 verify_opencv.py

# Hoặc dùng Makefile
make verify
```

### Bước 3: Cài thư viện Python của project

```bash
pip install -r requirements.txt
```

Lưu ý: OpenCV có SURF vẫn cần build từ source (đã xử lý ở Bước 1).

### Bước 4: Chạy chương trình

```bash
# Chế độ CLI: so khớp 2 ảnh
python main.py image1.jpg image2.jpg

# Ví dụ với ảnh trong data/
python main.py data/messi_1.jpg data/messi_2.jpg

# Chế độ UI: mở giao diện Gradio
python main.py
```

## 📋 Các Lệnh Makefile Hữu Ích

```bash
make help           # Hiển thị trợ giúp
make check-pyenv    # Kiểm tra pyenv setup
make download-opencv # Chỉ download source
make configure      # Chỉ configure CMake
make build          # Chỉ build
make install        # Chỉ install
make verify         # Kiểm tra installation
make rebuild        # Rebuild nhanh (không configure lại)
make clean          # Xóa build directory
make clean-all      # Xóa toàn bộ source và build
make show-log       # Xem build log
```

## 🔧 Troubleshooting

### Lỗi: CMake không tìm thấy Python

```bash
# Kiểm tra pyenv
make check-pyenv

# Nếu có lỗi, activate pyenv
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Thử lại
make check-pyenv
```

### Lỗi: SURF không hoạt động

```bash
# Kiểm tra chi tiết
python3 verify_opencv.py

# Nếu lỗi, xem build log
make show-log

# Rebuild
make clean
make configure
make build
make install
```

### Lỗi build (error trong quá trình make)

```bash
# Xem log chi tiết
cat ~/opencv_build/opencv/build/build.log | grep -i error

# Clean và rebuild
make clean
make build
```

## 📖 Tài liệu đầy đủ

- [README.md](README.md) - Mô tả dự án và cách sử dụng
- [OPENCV_PYENV_SETUP.md](OPENCV_PYENV_SETUP.md) - Hướng dẫn build thủ công chi tiết
- [Makefile](Makefile) - Automation script

## 💡 Tips

1. **Build lần đầu mất 30-60 phút** - Kiên nhẫn chờ đợi
2. **Dùng `make rebuild`** nếu chỉ sửa code OpenCV nhỏ
3. **Dùng `make verify`** sau mỗi lần install
4. **Kiểm tra `~/opencv_build/opencv/build/build.log`** nếu có lỗi

## 🎯 Output Files

Sau khi chạy `python main.py image1.jpg image2.jpg`:

```
outputs/
├── image1_surf_info.txt           # Keypoints info ảnh 1
├── image2_surf_info.txt           # Keypoints info ảnh 2
└── image1_vs_image2_surf_match.jpg # Ảnh kết quả (3 panels)
```

## ⚡ Workflow Thông thường

```bash
# Lần đầu tiên
make all                    # Setup toàn bộ

# Mỗi lần sử dụng
pip install -r requirements.txt    # Chỉ cần chạy khi chưa cài deps

# Chạy CLI
python main.py img1.jpg img2.jpg

# Hoặc chạy UI
python main.py --ui

# Nếu cần rebuild OpenCV
make rebuild

# Nếu gặp vấn đề
make verify
python3 verify_opencv.py
```

---

**Câu hỏi?** Xem [OPENCV_PYENV_SETUP.md](OPENCV_PYENV_SETUP.md) để biết chi tiết hoặc chạy `make help`.
