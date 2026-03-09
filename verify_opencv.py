#!/usr/bin/env python3
"""
Script kiểm tra OpenCV với SURF support
"""

import sys


def check_opencv():
    """Kiểm tra OpenCV installation"""
    print("=" * 50)
    print("Kiểm tra OpenCV Installation")
    print("=" * 50)
    print()

    # Test 1: Import cv2
    print("Test 1: Import OpenCV...")
    try:
        import cv2
        print(f"  ✓ OpenCV version: {cv2.__version__}")
        print(f"  ✓ Location: {cv2.__file__}")
    except ImportError as e:
        print(f"  ❌ Không thể import cv2: {e}")
        return False

    print()

    # Test 2: Check xfeatures2d module
    print("Test 2: Check xfeatures2d module...")
    try:
        import cv2.xfeatures2d
        print(f"  ✓ xfeatures2d available")
        print(f"  ✓ Location: {cv2.xfeatures2d.__file__}")
    except (ImportError, AttributeError) as e:
        print(f"  ❌ xfeatures2d không khả dụng: {e}")
        print("  → OpenCV cần được build với opencv_contrib")
        return False

    print()

    # Test 3: Create SURF detector
    print("Test 3: Create SURF detector...")
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        print(f"  ✓ SURF detector created successfully")
        print(f"  ✓ Hessian threshold: {surf.getHessianThreshold()}")
    except (AttributeError, cv2.error) as e:
        print(f"  ❌ Không thể tạo SURF detector: {e}")
        print("  → OpenCV cần build với OPENCV_ENABLE_NONFREE=ON")
        return False

    print()

    # Test 4: Check build info
    print("Test 4: Check build configuration...")
    build_info = cv2.getBuildInformation()

    # Check for nonfree
    if "OPENCV_ENABLE_NONFREE" in build_info or "Non-free algorithms" in build_info:
        print("  ✓ Non-free algorithms enabled")
    else:
        print("  ⚠ Warning: Không tìm thấy thông tin nonfree trong build info")

    # Check Python paths
    if "Python 3:" in build_info:
        for line in build_info.split('\n'):
            if 'Python 3:' in line or 'Python (for build)' in line:
                print(f"  {line.strip()}")

    print()

    # Test 5: Test với ảnh mẫu nhỏ
    print("Test 5: Test detection với ảnh mẫu...")
    try:
        import numpy as np

        # Tạo ảnh test 100x100 với gradient
        img = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            img[i, :] = i * 255 // 100

        # Thêm một số điểm nổi bật
        cv2.circle(img, (25, 25), 10, 255, -1)
        cv2.circle(img, (75, 75), 10, 255, -1)

        # Thử phát hiện keypoints
        surf = cv2.xfeatures2d.SURF_create(100)  # Threshold thấp để dễ detect
        keypoints, descriptors = surf.detectAndCompute(img, None)

        print(f"  ✓ Detected {len(keypoints)} keypoints trong ảnh test")
        if descriptors is not None:
            print(f"  ✓ Descriptor shape: {descriptors.shape}")

    except Exception as e:
        print(f"  ❌ Lỗi khi test detection: {e}")
        return False

    print()
    print("=" * 50)
    print("✓ TẤT CẢ TESTS PASSED!")
    print("OpenCV với SURF đã sẵn sàng sử dụng.")
    print("=" * 50)
    print()
    print("Bạn có thể chạy chương trình chính:")
    print("  python main.py image1.jpg image2.jpg")

    return True


def main():
    success = check_opencv()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
