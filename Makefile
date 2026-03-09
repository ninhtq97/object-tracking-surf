# Makefile for OpenCV Build with Pyenv
# Tự động hóa quá trình build OpenCV với SURF support cho pyenv environment

.PHONY: help check-pyenv install-deps download-opencv configure build install verify clean clean-all

# Màu sắc cho output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m # No Color

# Biến môi trường
BUILD_DIR := $(HOME)/opencv_build
OPENCV_DIR := $(BUILD_DIR)/opencv
CONTRIB_DIR := $(BUILD_DIR)/opencv_contrib
BUILD_TARGET := $(OPENCV_DIR)/build

# Python thông tin từ pyenv
PYTHON_EXECUTABLE := $(shell which python3)
PYTHON_VERSION := $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_PREFIX := $(shell python3 -c "import sys; print(sys.prefix)")
PYTHON_INCLUDE := $(PYTHON_PREFIX)/include/python$(PYTHON_VERSION)
PYTHON_SITE := $(shell python3 -c "import site; print(site.getsitepackages()[0])")
NCPU := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

help: ## Hiển thị trợ giúp
	@echo "$(GREEN)=== OpenCV Build with Pyenv - Makefile Help ===$(NC)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Quick Start:$(NC)"
	@echo "  make all          # Chạy toàn bộ quy trình (check -> download -> build -> install)"
	@echo "  make verify       # Kiểm tra sau khi install"

check-pyenv: ## Kiểm tra pyenv setup
	@echo "$(GREEN)=== Kiểm tra Pyenv Setup ===$(NC)"
	@echo "Python executable: $(PYTHON_EXECUTABLE)"
	@echo "Python version:    $(PYTHON_VERSION)"
	@echo "Python prefix:     $(PYTHON_PREFIX)"
	@echo "Python include:    $(PYTHON_INCLUDE)"
	@echo "Python site-pkg:   $(PYTHON_SITE)"
	@echo "CPU cores:         $(NCPU)"
	@echo ""
	@if [ ! -d "$(PYTHON_PREFIX)" ]; then \
		echo "$(RED)❌ Python prefix không tồn tại: $(PYTHON_PREFIX)$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(PYTHON_EXECUTABLE)" ]; then \
		echo "$(RED)❌ Python executable không tìm thấy: $(PYTHON_EXECUTABLE)$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Pyenv setup OK$(NC)"

install-deps: ## Cài đặt dependencies (Homebrew/apt)
	@echo "$(GREEN)=== Cài đặt Dependencies ===$(NC)"
	@if command -v brew >/dev/null 2>&1; then \
		echo "$(YELLOW)Đang cài đặt dependencies với Homebrew...$(NC)"; \
		brew install cmake pkg-config jpeg libpng libtiff openexr eigen tbb hdf5; \
	elif command -v apt-get >/dev/null 2>&1; then \
		echo "$(YELLOW)Đang cài đặt dependencies với apt...$(NC)"; \
		sudo apt-get update; \
		sudo apt-get install -y build-essential cmake pkg-config \
			libjpeg-dev libpng-dev libtiff-dev \
			libeigen3-dev libtbb-dev libhdf5-dev; \
	else \
		echo "$(RED)❌ Không tìm thấy package manager (brew/apt)$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Dependencies đã cài đặt$(NC)"

download-opencv: ## Download OpenCV và opencv_contrib
	@echo "$(GREEN)=== Download OpenCV Source ===$(NC)"
	@mkdir -p $(BUILD_DIR)
	@if [ ! -d "$(OPENCV_DIR)" ]; then \
		echo "$(YELLOW)Cloning opencv...$(NC)"; \
		cd $(BUILD_DIR) && git clone https://github.com/opencv/opencv.git; \
		cd $(OPENCV_DIR) && git checkout 4.x; \
	else \
		echo "$(YELLOW)OpenCV đã tồn tại, đang pull updates...$(NC)"; \
		cd $(OPENCV_DIR) && git pull; \
	fi
	@if [ ! -d "$(CONTRIB_DIR)" ]; then \
		echo "$(YELLOW)Cloning opencv_contrib...$(NC)"; \
		cd $(BUILD_DIR) && git clone https://github.com/opencv/opencv_contrib.git; \
		cd $(CONTRIB_DIR) && git checkout 4.x; \
	else \
		echo "$(YELLOW)opencv_contrib đã tồn tại, đang pull updates...$(NC)"; \
		cd $(CONTRIB_DIR) && git pull; \
	fi
	@echo "$(GREEN)✓ Download hoàn tất$(NC)"

configure: check-pyenv ## Configure CMake với pyenv settings
	@echo "$(GREEN)=== Configure CMake ===$(NC)"
	@mkdir -p $(BUILD_TARGET)
	@cd $(BUILD_TARGET) && \
		cmake \
		-D CMAKE_BUILD_TYPE=Release \
		-D CMAKE_INSTALL_PREFIX=$(PYTHON_PREFIX) \
		-D OPENCV_ENABLE_NONFREE=ON \
		-D OPENCV_EXTRA_MODULES_PATH=$(CONTRIB_DIR)/modules \
		-D PYTHON3_EXECUTABLE=$(PYTHON_EXECUTABLE) \
		-D PYTHON3_INCLUDE_DIR=$(PYTHON_INCLUDE) \
		-D PYTHON3_LIBRARY=$(PYTHON_PREFIX)/lib/libpython$(PYTHON_VERSION).dylib \
		-D PYTHON3_PACKAGES_PATH=$(PYTHON_SITE) \
		-D BUILD_opencv_python3=ON \
		-D BUILD_opencv_contrib_python3=ON \
		-D BUILD_EXAMPLES=OFF \
		-D BUILD_TESTS=OFF \
		-D BUILD_PERF_TESTS=OFF \
		-D WITH_TBB=ON \
		-D WITH_EIGEN=ON \
		-D OPENCV_GENERATE_PKGCONFIG=ON \
		..
	@echo ""
	@echo "$(YELLOW)=== Checking CMake Configuration ===$(NC)"
	@cd $(BUILD_TARGET) && cmake -LA | grep -E "PYTHON|NONFREE" || true
	@echo "$(GREEN)✓ Configure hoàn tất$(NC)"

build: ## Build OpenCV (có thể mất 30-60 phút)
	@echo "$(GREEN)=== Building OpenCV với $(NCPU) cores ===$(NC)"
	@echo "$(YELLOW)⚠ Quá trình này có thể mất 30-60 phút...$(NC)"
	@cd $(BUILD_TARGET) && make -j$(NCPU) 2>&1 | tee build.log
	@echo ""
	@echo "$(YELLOW)Kiểm tra errors trong build.log...$(NC)"
	@if grep -qi "error" $(BUILD_TARGET)/build.log; then \
		echo "$(RED)❌ Có lỗi trong quá trình build. Kiểm tra $(BUILD_TARGET)/build.log$(NC)"; \
		grep -i "error" $(BUILD_TARGET)/build.log | head -10; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Build thành công$(NC)"

install: ## Install OpenCV vào pyenv environment
	@echo "$(GREEN)=== Installing OpenCV ===$(NC)"
	@cd $(BUILD_TARGET) && make install
	@echo "$(GREEN)✓ Install hoàn tất$(NC)"
	@echo ""
	@echo "$(YELLOW)Chạy 'make verify' để kiểm tra installation$(NC)"

verify: ## Kiểm tra OpenCV đã cài đặt đúng
	@echo "$(GREEN)=== Verify OpenCV Installation ===$(NC)"
	@echo ""
	@echo "Test 1: Check version"
	@python3 -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || \
		(echo "$(RED)  ❌ Không thể import cv2$(NC)" && exit 1)
	@echo ""
	@echo "Test 2: Check location"
	@python3 -c "import cv2; print(f'  ✓ Location: {cv2.__file__}')"
	@echo ""
	@echo "Test 3: Check xfeatures2d module"
	@python3 -c "import cv2.xfeatures2d; print('  ✓ xfeatures2d available')" || \
		(echo "$(RED)  ❌ xfeatures2d không khả dụng$(NC)" && exit 1)
	@echo ""
	@echo "Test 4: Check SURF"
	@python3 -c "import cv2; surf = cv2.xfeatures2d.SURF_create(); print('  ✓ SURF works!')" || \
		(echo "$(RED)  ❌ SURF không hoạt động$(NC)" && exit 1)
	@echo ""
	@echo "$(GREEN)✓ Tất cả tests passed! OpenCV với SURF đã sẵn sàng.$(NC)"

clean: ## Xóa build directory
	@echo "$(YELLOW)Xóa build directory: $(BUILD_TARGET)$(NC)"
	@rm -rf $(BUILD_TARGET)
	@echo "$(GREEN)✓ Đã xóa build directory$(NC)"

clean-all: ## Xóa toàn bộ opencv source và build
	@echo "$(RED)⚠ CẢNH BÁO: Sẽ xóa toàn bộ $(BUILD_DIR)$(NC)"
	@read -p "Bạn có chắc chắn? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf $(BUILD_DIR); \
		echo "$(GREEN)✓ Đã xóa toàn bộ$(NC)"; \
	else \
		echo "$(YELLOW)Hủy bỏ$(NC)"; \
	fi

all: check-pyenv install-deps download-opencv configure build install verify ## Chạy toàn bộ quy trình
	@echo ""
	@echo "$(GREEN)╔═══════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║   ✓ HOÀN TẤT CÀI ĐẶT OPENCV VỚI SURF    ║$(NC)"
	@echo "$(GREEN)╚═══════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "Bạn có thể chạy chương trình bằng:"
	@echo "  $(YELLOW)python main.py image1.jpg image2.jpg$(NC)"

# Target riêng cho WSL/Linux (thay đổi NCPU command)
configure-linux: PYTHON_LIBRARY := $(PYTHON_PREFIX)/lib/libpython$(PYTHON_VERSION).so
configure-linux: NCPU := $(shell nproc 2>/dev/null || echo 4)
configure-linux: configure

# Target riêng để rebuild nhanh (không configure lại)
rebuild: ## Rebuild nhanh (không configure lại)
	@$(MAKE) build install

# Target để xem build log
show-log: ## Hiển thị build log
	@if [ -f "$(BUILD_TARGET)/build.log" ]; then \
		less $(BUILD_TARGET)/build.log; \
	else \
		echo "$(RED)Không tìm thấy build.log$(NC)"; \
	fi
