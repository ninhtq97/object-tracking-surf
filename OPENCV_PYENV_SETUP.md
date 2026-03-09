# OpenCV Build với Pyenv - Hướng dẫn Chi tiết

## Vấn đề với Pyenv

Khi dùng pyenv, Python được cài vào `/Users/<user>/.pyenv/versions/<version>/` thay vì `/usr/local/`. Điều này gây vấn đề:

1. **CMake không tìm được Python paths đúng**
2. **Python bindings không install vào đúng site-packages**
3. **Built library conflict với system Python**

## Kiểm tra Pyenv Setup

```bash
# Kiểm tra Python hiện tại
which python3
# Output: /Users/<user>/.pyenv/shims/python3

python3 --version
# Output: Python 3.10.x (hoặc version khác)

# Kiểm tra Python thực tế từ pyenv
echo $PYENV_VERSION
pyenv versions
# Xem dòng * để biết version active
```

## Giải pháp đầy đủ cho Pyenv

### Bước 1: Chuẩn bị

```bash
# Kiểm tra pyenv virtual environment
python3 -c "import sys; print(sys.prefix)"
# Output: /Users/<user>/.pyenv/versions/3.10.x

# Kiểm tra site-packages
python3 -c "import site; print(site.getsitepackages())"
# Output: ['/Users/<user>/.pyenv/versions/3.10.x/lib/python3.10/site-packages', ...]

# Lưu những thông tin này - cần dùng lại
PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
PYTHON_INCLUDE="${PYTHON_PREFIX}/include/python3.10"  # Đổi 3.10 thành version của bạn
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "Python prefix: $PYTHON_PREFIX"
echo "Python include: $PYTHON_INCLUDE"
echo "Python site-packages: $PYTHON_SITE"
```

### Bước 2: Cài dependencies

```bash
# Cài với Homebrew (không phải pyenv Python)
/usr/bin/python3 -m pip install --upgrade pip  # Dùng system Python
brew install cmake pkg-config jpeg libpng libtiff openexr eigen tbb hdf5
```

### Bước 3: Download OpenCV

```bash
mkdir ~/opencv_build && cd ~/opencv_build

# Clone OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x
cd ..

# Clone opencv_contrib
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.x
cd ..
```

### Bước 4: Configure CMake (quan trọng cho Pyenv)

```bash
cd ~/opencv_build/opencv

# Xóa build cũ nếu có
rm -rf build
mkdir build && cd build

# Lấy Python info từ pyenv
PYTHON_EXECUTABLE=$(which python3)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
PYTHON_INCLUDE="${PYTHON_PREFIX}/include/python${PYTHON_VERSION}"
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "=== CMake Configuration for Pyenv ==="
echo "Python executable: $PYTHON_EXECUTABLE"
echo "Python version: $PYTHON_VERSION"
echo "Python prefix: $PYTHON_PREFIX"
echo "Python include: $PYTHON_INCLUDE"
echo "Python site-packages: $PYTHON_SITE"
echo ""

# Run CMake với Pyenv-specific settings
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$PYTHON_PREFIX \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D PYTHON3_EXECUTABLE=$PYTHON_EXECUTABLE \
  -D PYTHON3_INCLUDE_DIR=$PYTHON_INCLUDE \
  -D PYTHON3_LIBRARY="${PYTHON_PREFIX}/lib/libpython${PYTHON_VERSION}.dylib" \
  -D PYTHON3_PACKAGES_PATH=$PYTHON_SITE \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_contrib_python3=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D WITH_TBB=ON \
  -D WITH_EIGEN=ON \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  ..

# Kiểm tra CMake configuration
cmake -LA | grep -E "PYTHON|NONFREE"
# Phải thấy:
# OPENCV_ENABLE_NONFREE:BOOL=ON
# PYTHON3_EXECUTABLE:FILEPATH=$PYTHON_EXECUTABLE
# PYTHON3_INCLUDE_DIR:PATH=$PYTHON_INCLUDE
# PYTHON3_PACKAGES_PATH:PATH=$PYTHON_SITE
```

**Quan trọng:** Phải để `CMAKE_INSTALL_PREFIX=$PYTHON_PREFIX` để install vào pyenv environment.

### Bước 5: Build

```bash
cd ~/opencv_build/opencv/build

# Build (có thể mất 30-60 phút)
NCPU=$(sysctl -n hw.ncpu)
echo "Building with $NCPU cores..."
make -j$NCPU 2>&1 | tee build.log

# Kiểm tra có error không
grep -i "error\|failed" build.log | head -10
```

### Bước 6: Install

```bash
cd ~/opencv_build/opencv/build

# Install vào pyenv environment (không cần sudo)
make install

# Nếu cần: make reinstall
```

### Bước 7: Verify

```bash
# Test 1: Check version
python3 -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# Test 2: Check location
python3 -c "import cv2; print(cv2.__file__)"
# Output phải chứa: .pyenv/versions/...

# Test 3: Check xfeatures2d
python3 -c "import cv2.xfeatures2d; print(cv2.xfeatures2d.__file__)"

# Test 4: Check SURF
python3 -c "import cv2; surf = cv2.xfeatures2d.SURF_create(); print('✓ SURF works!')"

# Test 5: Run diagnostic
python3 diagnose_opencv.py
```

## Xử lý sự cố với Pyenv

### Problem: Python path sai

```bash
# Kiểm tra
which python3
# Nếu không phải /Users/<user>/.pyenv/shims/python3, activate pyenv

# Activate pyenv shell
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Set version
pyenv local 3.10.x  # hoặc version bạn dùng
which python3  # Phải là .pyenv/shims/python3 bây giờ
```

### Problem: CMake không tìm Python

```bash
# Full path CMake
cd ~/opencv_build/opencv/build
rm CMakeCache.txt

# Lấy đầy đủ info
PYTHON_EXECUTABLE=$(which python3)
PYTHON_CONFIG=$(python3-config --prefix)

# Kiểm tra file exist
ls -la "$PYTHON_EXECUTABLE"
ls -la "$(python3-config --includes | sed 's/-I//g')"

# Re-run CMake với absolute paths
cmake \
  -D PYTHON3_EXECUTABLE="$PYTHON_EXECUTABLE" \
  -D PYTHON3_INCLUDE_DIR="$(python3 -c "from sysconfig import get_path; print(get_path('include'))")" \
  -D PYTHON3_LIBRARY="$(python3-config --prefix)/lib/libpython3.10.dylib" \
  -D PYTHON3_PACKAGES_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
  ..
```

### Problem: Conflict với system Python

```bash
# Ensure pyenv không bị conflict
pyenv versions
# * 3.10.x (set by /path/to/.python-version)
# system

# Remove old OpenCV from system
/usr/bin/python3 -m pip uninstall opencv-python opencv-contrib-python -y

# Confirm pyenv version đang dùng
python3 --version
which python3
```

### Problem: Build failed - Library mismatch

```bash
# Full clean rebuild
rm -rf ~/opencv_build/opencv/build

# Đảm bảo pyenv environment clean
python3 -m pip uninstall opencv-python -y
python3 -m pip cache purge

# Re-run CMake + build
mkdir ~/opencv_build/opencv/build && cd ~/opencv_build/opencv/build
cmake ... (với flags đầy đủ ở trên)
make -j8
make install
```

## Script Helper cho Pyenv

```bash
#!/bin/bash
# setup_opencv_pyenv.sh

set -e

echo "=== Setting up OpenCV with Pyenv ==="

# Get Python info
PYTHON_EXECUTABLE=$(which python3)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)")
PYTHON_INCLUDE="${PYTHON_PREFIX}/include/python${PYTHON_VERSION}"
PYTHON_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
NCPU=$(sysctl -n hw.ncpu)

echo "Python: $PYTHON_EXECUTABLE"
echo "Version: $PYTHON_VERSION"
echo "Prefix: $PYTHON_PREFIX"
echo ""

# Prepare
mkdir -p ~/opencv_build && cd ~/opencv_build

if [ ! -d "opencv" ]; then
    git clone https://github.com/opencv/opencv.git
    cd opencv && git checkout 4.x && cd ..
fi

if [ ! -d "opencv_contrib" ]; then
    git clone https://github.com/opencv/opencv_contrib.git
    cd opencv_contrib && git checkout 4.x && cd ..
fi

# Build
mkdir -p opencv/build && cd opencv/build
rm -f CMakeCache.txt

cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$PYTHON_PREFIX \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D PYTHON3_EXECUTABLE=$PYTHON_EXECUTABLE \
  -D PYTHON3_INCLUDE_DIR=$PYTHON_INCLUDE \
  -D PYTHON3_PACKAGES_PATH=$PYTHON_SITE \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_contrib_python3=ON \
  -D WITH_TBB=ON \
  ..

make -j$NCPU
make install

echo "✓ OpenCV build complete!"
python3 -c "import cv2; print(f'OpenCV {cv2.__version__} installed')"
```

## Liên kết tham khảo

- Pyenv: https://github.com/pyenv/pyenv
- OpenCV Pyenv: https://github.com/opencv/opencv/wiki/Building-OpenCV-with-pyenv
- Python sysconfig: https://docs.python.org/3/library/sysconfig.html
