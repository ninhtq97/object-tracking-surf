"""
Object Detection using SURF (Speeded-Up Robust Features)
Phát hiện đặc trưng sử dụng SURF theo OpenCV docs
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def detect_surf_features(image_path, output_dir="outputs"):
    """
    Phát hiện đặc trưng SURF trong ảnh, trả về keypoints/descriptors và lưu file info.

    Args:
        image_path: Đường dẫn tới ảnh cần phát hiện
        output_dir: Thư mục lưu kết quả

    Returns:
        img: Ảnh gốc (BGR)
        keypoints: Danh sách các điểm đặc trưng phát hiện được
        descriptors: Descriptors tương ứng
        img_with_keypoints: Ảnh đã vẽ keypoints
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    # Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Khởi tạo SURF detector
    try:
        surf = cv2.xfeatures2d.SURF_create(400)  # hessian threshold = 400
        print(f"✓ Đã khởi tạo SURF detector")
    except (AttributeError, cv2.error) as e:
        raise RuntimeError(
            f"SURF không khả dụng trong OpenCV hiện tại.\n"
            f"Vui lòng build OpenCV với OPENCV_ENABLE_NONFREE=ON.\n"
            f"Chi tiết: {str(e)}"
        )

    # Phát hiện keypoints và tính descriptors
    keypoints, descriptors = surf.detectAndCompute(gray, None)

    print(f"Đã phát hiện {len(keypoints)} keypoints")

    # Vẽ keypoints lên ảnh
    img_with_keypoints = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Tạo thư mục output nếu chưa có (dùng cho file info)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Lưu thông tin keypoints vào file text
    img_name = Path(image_path).stem
    info_file = output_path / f"{img_name}_surf_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Ảnh: {image_path}\n")
        f.write(f"Số keypoints: {len(keypoints)}\n\n")
        f.write("Chi tiết keypoints (10 điểm đầu):\n")
        for i, kp in enumerate(keypoints):
            f.write(f"  {i+1}. Tọa độ: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f}), "
                   f"Size: {kp.size:.2f}, Angle: {kp.angle:.2f}°, "
                   f"Response: {kp.response:.4f}\n")

    print(f"✓ Đã lưu thông tin: {info_file}")

    return img, keypoints, descriptors, img_with_keypoints


def match_two_images(image1_path, image2_path, output_dir="outputs"):
    """
    Phát hiện SURF trên hai ảnh và vẽ đường nối keypoint so khớp giữa chúng.
    """
    img1, kp1, des1, img1_with = detect_surf_features(image1_path, output_dir)
    img2, kp2, des2, img2_with = detect_surf_features(image2_path, output_dir)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        raise RuntimeError("Không đủ keypoints để so khớp giữa hai ảnh")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:min(100, len(matches))]

    # Vẽ match trên ảnh gốc (không có keypoints vẽ sẵn) để đảm bảo đường nối chính xác
    matched_vis = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Ghép 3 panel theo chiều dọc: ảnh1_detected, ảnh2_detected, match
    def pad_to_width(img, target_w):
        h, w = img.shape[:2]
        if w == target_w:
            return img
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        return cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    target_w = max(img1_with.shape[1], img2_with.shape[1], matched_vis.shape[1])
    img1_panel = pad_to_width(img1_with, target_w)
    img2_panel = pad_to_width(img2_with, target_w)
    match_panel = pad_to_width(matched_vis, target_w)
    composite = np.vstack([img1_panel, img2_panel, match_panel])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img1_name = Path(image1_path).stem
    img2_name = Path(image2_path).stem
    match_file = output_path / f"{img1_name}_vs_{img2_name}_surf_match.jpg"
    cv2.imwrite(str(match_file), composite)
    print(f"✓ Đã lưu ảnh so khớp giữa hai ảnh: {match_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="So khớp đặc trưng SURF giữa 2 ảnh"
    )
    parser.add_argument(
        "image1",
        type=str,
        help="Ảnh gốc/tham chiếu"
    )
    parser.add_argument(
        "image2",
        type=str,
        help="Ảnh cần so khớp"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Thư mục lưu kết quả (mặc định: outputs)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== SURF Feature Matching giữa 2 ảnh ===")
    print(f"Ảnh 1: {args.image1}")
    print(f"Ảnh 2: {args.image2}")
    print(f"Thư mục output: {args.output_dir}")
    print()

    try:
        match_two_images(args.image1, args.image2, args.output_dir)
        print("=== Hoàn tất ===")
        return 0
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
