"""
Object Detection using SURF (Speeded-Up Robust Features)
Phát hiện đặc trưng sử dụng SURF theo OpenCV docs
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def _create_surf_detector():
    """Khởi tạo SURF detector với kiểm tra khả dụng của OpenCV nonfree."""
    try:
        surf = cv2.xfeatures2d.SURF_create(400)  # hessian threshold = 400
        return surf
    except (AttributeError, cv2.error) as e:
        raise RuntimeError(
            "SURF không khả dụng trong OpenCV hiện tại.\n"
            "Vui lòng build OpenCV với OPENCV_ENABLE_NONFREE=ON.\n"
            f"Chi tiết: {str(e)}"
        )


def _detect_surf_from_bgr_image(
    img: np.ndarray,
    image_label: str,
    output_dir: str = "outputs",
) -> Tuple[list, np.ndarray, np.ndarray]:
    """
    Phát hiện keypoints/descriptors từ ảnh BGR và lưu file thông tin.
    """
    if img is None or img.size == 0:
        raise ValueError(f"Ảnh rỗng hoặc không hợp lệ: {image_label}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = _create_surf_detector()
    print("✓ Đã khởi tạo SURF detector")

    keypoints, descriptors = surf.detectAndCompute(gray, None)
    print(f"Đã phát hiện {len(keypoints)} keypoints cho {image_label}")

    img_with_keypoints = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    info_file = output_path / f"{image_label}_surf_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Ảnh: {image_label}\n")
        f.write(f"Số keypoints: {len(keypoints)}\n\n")
        f.write("Chi tiết keypoints:\n")
        for i, kp in enumerate(keypoints):
            f.write(
                f"  {i+1}. Tọa độ: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f}), "
                f"Size: {kp.size:.2f}, Angle: {kp.angle:.2f}°, "
                f"Response: {kp.response:.4f}\n"
            )

    print(f"✓ Đã lưu thông tin: {info_file}")
    return keypoints, descriptors, img_with_keypoints


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

    img_name = Path(image_path).stem
    keypoints, descriptors, img_with_keypoints = _detect_surf_from_bgr_image(
        img=img,
        image_label=img_name,
        output_dir=output_dir,
    )

    return img, keypoints, descriptors, img_with_keypoints


def _match_and_render(
    img1: np.ndarray,
    img2: np.ndarray,
    name1: str,
    name2: str,
    output_dir: str = "outputs",
):
    """So khớp SURF giữa 2 ảnh BGR và trả về ảnh composite kết quả."""
    kp1, des1, img1_with = _detect_surf_from_bgr_image(img1, name1, output_dir)
    kp2, des2, img2_with = _detect_surf_from_bgr_image(img2, name2, output_dir)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        raise RuntimeError("Không đủ keypoints để so khớp giữa hai ảnh")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:min(100, len(matches))]

    matched_vis = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    def pad_to_width(img, target_w):
        h, w = img.shape[:2]
        if w == target_w:
            return img
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        return cv2.copyMakeBorder(
            img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    target_w = max(img1_with.shape[1], img2_with.shape[1], matched_vis.shape[1])
    img1_panel = pad_to_width(img1_with, target_w)
    img2_panel = pad_to_width(img2_with, target_w)
    match_panel = pad_to_width(matched_vis, target_w)
    composite = np.vstack([img1_panel, img2_panel, match_panel])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    match_file = output_path / f"{name1}_vs_{name2}_surf_match.jpg"
    cv2.imwrite(str(match_file), composite)
    print(f"✓ Đã lưu ảnh so khớp giữa hai ảnh: {match_file}")

    return composite, match_file, len(kp1), len(kp2), len(matches)


def match_two_images(image1_path, image2_path, output_dir="outputs"):
    """
    Phát hiện SURF trên hai ảnh và vẽ đường nối keypoint so khớp giữa chúng.
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image1_path}")
    if img2 is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image2_path}")

    img1_name = Path(image1_path).stem
    img2_name = Path(image2_path).stem
    _match_and_render(
        img1=img1,
        img2=img2,
        name1=img1_name,
        name2=img2_name,
        output_dir=output_dir,
    )


def launch_ui(output_dir="outputs"):
    """Khởi chạy giao diện web Gradio để chọn 2 ảnh và xem kết quả SURF."""
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError(
            "Chưa cài gradio. Hãy cài bằng lệnh: pip install gradio"
        )

    def on_submit(image1, image2):
        if image1 is None or image2 is None:
            raise gr.Error("Vui lòng chọn đủ 2 ảnh trước khi submit")

        try:
            img1_bgr = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
            img2_bgr = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

            composite_bgr, match_file, kp1_count, kp2_count, match_count = _match_and_render(
                img1=img1_bgr,
                img2=img2_bgr,
                name1="image_1",
                name2="image_2",
                output_dir=output_dir,
            )
            composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
            message = (
                f"Hoàn tất. Keypoints ảnh 1: {kp1_count}, "
                f"ảnh 2: {kp2_count}, số match hiển thị: {match_count}. "
                f"Đã lưu: {match_file}"
            )
            return composite_rgb, message
        except Exception as e:
            raise gr.Error(str(e))

    custom_css = """
    .gradio-container {
        max-width: 1100px !important;
        margin: 0 auto;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4ecf7 100%);
    }
    .block-title {
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .block-subtitle {
        text-align: center;
        color: #3c4a63;
        margin-bottom: 20px;
    }
    .panel-card {
        border-radius: 16px;
        border: 1px solid #d7e1ef;
        box-shadow: 0 8px 24px rgba(13, 38, 76, 0.08);
    }
    .submit-row {
        display: flex;
        justify-content: center;
    }
    """

    with gr.Blocks(title="SURF Feature Matching") as demo:
        gr.HTML("<div class='block-title'>SURF Feature Matching</div>")
        gr.HTML("<div class='block-subtitle'>Chọn 2 ảnh, trích xuất đặc trưng và xem kết quả so khớp ngay</div>")

        with gr.Row(equal_height=True):
            with gr.Column(elem_classes=["panel-card"]):
                gr.Markdown("### Ảnh 1")
                image1_input = gr.Image(
                    label="Select ảnh 1",
                    type="numpy",
                    sources=["upload"],
                    height=280,
                )

            with gr.Column(elem_classes=["panel-card"]):
                gr.Markdown("### Ảnh 2")
                image2_input = gr.Image(
                    label="Select ảnh 2",
                    type="numpy",
                    sources=["upload"],
                    height=280,
                )

        with gr.Row(elem_classes=["submit-row"]):
            submit_btn = gr.Button(
                "Trích xuất đặc trưng 2 ảnh",
                variant="primary",
                size="lg",
            )

        with gr.Row():
            result_image = gr.Image(
                label="Kết quả sau khi trích xuất",
                type="numpy",
                height=520,
                elem_classes=["panel-card"],
            )

        status_box = gr.Textbox(label="Trạng thái", interactive=False)

        submit_btn.click(
            fn=on_submit,
            inputs=[image1_input, image2_input],
            outputs=[result_image, status_box],
        )

    demo.launch(theme=gr.themes.Soft(), css=custom_css)


def parse_args():
    parser = argparse.ArgumentParser(
        description="So khớp đặc trưng SURF giữa 2 ảnh (CLI hoặc UI)"
    )
    parser.add_argument(
        "image1",
        type=str,
        nargs="?",
        help="Ảnh gốc/tham chiếu"
    )
    parser.add_argument(
        "image2",
        type=str,
        nargs="?",
        help="Ảnh cần so khớp"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Thư mục lưu kết quả (mặc định: outputs)"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Khởi chạy giao diện người dùng Gradio"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ui or (args.image1 is None or args.image2 is None):
        print("=== Khởi chạy giao diện SURF ===")
        print(f"Thư mục output: {args.output_dir}")
        try:
            launch_ui(args.output_dir)
            return 0
        except Exception as e:
            print(f"❌ Lỗi khi chạy UI: {e}")
            return 1

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
