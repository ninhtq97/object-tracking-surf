import sys
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


def create_tracker():
    """Tạo SURF-based tracker sử dụng ORB features + matching"""
    return SurfTracker()


class SurfTracker:
    """Feature-based tracker sử dụng ORB (SURF alternative) + CSRT + Template Matching"""
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.akaze = cv2.AKAZE_create()
        # Use BFMatcher for binary descriptors (ORB/AKAZE), NOT FLANN
        self.matcher_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.detector = self.orb

        self.csrt_tracker = None
        self.use_csrt = False
        self.use_template = False
        self.last_error = ""

        self.roi_frame = None
        self.roi_kp = None
        self.roi_des = None
        self.roi_bbox = None
        self.roi_template = None
        self.initialized = False

        # Frame-to-frame smoothing and lost tracking handling
        self.last_valid_bbox = None
        self.consecutive_lost_frames = 0  # Counter for consecutive tracking failures

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def init(self, frame, bbox):
        """Khởi tạo tracker với ROI - robust multi-fallback initialization"""
        self.last_error = ""
        self.use_template = False
        self.use_csrt = False

        fh, fw = frame.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, min(fw - 1, x))
        y = max(0, min(fh - 1, y))
        w = max(1, min(fw - x, w))
        h = max(1, min(fh - y, h))

        if w < 8 or h < 8:
            self.last_error = "ROI quá nhỏ (< 8x8)"
            return False

        self.roi_bbox = (x, y, w, h)
        self.roi_frame = frame[y:y+h, x:x+w].copy()

        if self.roi_frame.size == 0:
            self.last_error = "ROI rỗng"
            return False

        # Lưu template cho fallback matching
        self.roi_template = cv2.cvtColor(self.roi_frame, cv2.COLOR_BGR2GRAY)

        # Try feature-based detection với multiple preprocessing
        roi_gray = cv2.cvtColor(self.roi_frame, cv2.COLOR_BGR2GRAY)

        # Normalize histogram để robust với brightness changes
        roi_normalized = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)

        # Strategy 1: CLAHE + denoising
        roi_prep = self.clahe.apply(roi_normalized)
        roi_prep = cv2.medianBlur(roi_prep, 3)

        kp, des = self._try_detect_features(roi_prep)
        detector_used = "ORB"

        # Strategy 2: Histogram equalization + AKAZE if ORB fails
        if des is None or len(kp) < 3:
            roi_eq = cv2.equalizeHist(roi_normalized)
            roi_eq = cv2.medianBlur(roi_eq, 3)
            kp, des = self._try_detect_akaze(roi_eq)
            detector_used = "AKAZE"

        # Strategy 3: Simple normalized grayscale
        if des is None or len(kp) < 3:
            kp, des = self.orb.detectAndCompute(roi_normalized, None)
            detector_used = "ORB+NORM"

        # Success with feature detection
        if des is not None and len(kp) >= 3:
            self.detector = self.orb if detector_used == "ORB" else self.akaze
            self.roi_kp = kp
            self.roi_des = des
            self.csrt_tracker = None
            self.use_csrt = False
            self.use_template = False
            self.initialized = True
            return True

        # Fallback 1: Try CSRT tracker
        csrt = self._create_csrt_tracker()
        if csrt is not None:
            try:
                if csrt.init(frame, (x, y, w, h)):
                    self.csrt_tracker = csrt
                    self.use_csrt = True
                    self.initialized = True
                    self.last_error = ""
                    return True
            except:
                pass

        # Fallback 2: Template Matching (ultra-robust for simple scenes)
        self.use_template = True
        self.initialized = True
        self.last_error = ""
        return True

    def _try_detect_features(self, roi_img):
        """Try ORB with relaxed settings"""
        try:
            kp, des = self.orb.detectAndCompute(roi_img, None)
            return kp, des
        except:
            return [], None

    def _try_detect_akaze(self, roi_img):
        """Try AKAZE as alternative detector"""
        try:
            kp, des = self.akaze.detectAndCompute(roi_img, None)
            return kp, des
        except:
            return [], None

    def update(self, frame):
        """Cập nhật vị trí đối tượng"""
        if not self.initialized:
            return False, self.roi_bbox

        # Template matching fallback (ultra-robust)
        if self.use_template:
            return self._update_template_match(frame)

        # CSRT fallback
        if self.use_csrt:
            if self.csrt_tracker is None:
                return False, self.roi_bbox
            try:
                ok, bbox = self.csrt_tracker.update(frame)
                if not ok:
                    return False, self.roi_bbox
                x, y, w, h = [int(v) for v in bbox]
                self.roi_bbox = (x, y, w, h)
                return True, self.roi_bbox
            except:
                return False, self.roi_bbox

        # Feature-based tracking
        if self.roi_des is None or len(self.roi_kp) < 3:
            return False, self.roi_bbox

        # Tìm features trong frame hiện tại với preprocessing giống ROI
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_normalized = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX)
        frame_prep = self.clahe.apply(frame_normalized)
        frame_prep = cv2.medianBlur(frame_prep, 3)

        frame_kp, frame_des = self.detector.detectAndCompute(frame_prep, None)

        if frame_des is None or len(frame_kp) < 3:
            return False, self.roi_bbox

        # Match features với Lowe's ratio test (robust matching)
        try:
            matches_raw = self.matcher_bf.knnMatch(self.roi_des, frame_des, k=2)
            if matches_raw is None or len(matches_raw) == 0:
                return False, self.roi_bbox

            # Apply Lowe's ratio test để filter outliers
            good_matches = []
            for pair in matches_raw:
                if len(pair) == 2:
                    m, n = pair
                    # Ratio threshold được hạ xuống từ 0.75 để robust hơn với brightness changes
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                elif len(pair) == 1:
                    # Nếu chỉ có 1 match (edge case), accept nếu distance nhỏ
                    if pair[0].distance < 50:
                        good_matches.append(pair[0])
        except:
            return False, self.roi_bbox

        # Cần ít nhất 3 matches để tính toán
        if len(good_matches) < 3:
            return False, self.roi_bbox

        # Extract matched points
        src_pts = cv2.KeyPoint_convert([self.roi_kp[m.queryIdx] for m in good_matches])
        dst_pts = cv2.KeyPoint_convert([frame_kp[m.trainIdx] for m in good_matches])

        src_pts = src_pts.reshape(-1, 1, 2).astype('float32')
        dst_pts = dst_pts.reshape(-1, 1, 2).astype('float32')

        # Tính homography matrix với RANSAC
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                return False, self.roi_bbox

            # Validate homography: kiểm tra inlier ratio
            if mask is not None:
                inlier_ratio = np.sum(mask) / len(mask)
                if inlier_ratio < 0.3:  # Quá ít inliers -> bad homography
                    return False, self.roi_bbox
        except:
            return False, self.roi_bbox

        # Transform ROI corners
        x, y, w, h = self.roi_bbox
        corners = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype='float32').reshape(-1, 1, 2)

        try:
            transformed = cv2.perspectiveTransform(corners, H)
            pts = transformed.reshape(-1, 2)

            # Tính bounding box từ transformed points
            left = int(max(0, np.min(pts[:, 0])))
            top = int(max(0, np.min(pts[:, 1])))
            right = int(np.max(pts[:, 0]))
            bottom = int(np.max(pts[:, 1]))

            new_w = max(1, right - left)
            new_h = max(1, bottom - top)

            # Sanity check: không cho phép bbox thay đổi quá many
            old_w, old_h = w, h
            size_change_ratio = (new_w * new_h) / (old_w * old_h)
            if size_change_ratio < 0.2 or size_change_ratio > 5.0:
                # Size thay đổi quá lớn -> likely wrong match
                return False, self.roi_bbox

            # Validate bbox changed smoothly (not jumping)
            if self.last_valid_bbox is not None:
                old_x, old_y, old_w, old_h = self.last_valid_bbox
                # Check if bbox movement is reasonable
                center_dist = abs(left + new_w//2 - (old_x + old_w//2)) + abs(top + new_h//2 - (old_y + old_h//2))
                if center_dist > max(old_w, old_h) * 2:
                    # Center jumped too far -> likely wrong match
                    return False, self.roi_bbox

            self.roi_bbox = (left, top, new_w, new_h)
            self.last_valid_bbox = self.roi_bbox
            self.consecutive_lost_frames = 0  # Reset lost counter on success
            return True, self.roi_bbox
        except:
            return False, self.roi_bbox

    def _update_template_match(self, frame):
        """Ultra-robust template matching fallback with strict validation"""
        if self.roi_template is None or self.roi_template.size == 0:
            return False, self.roi_bbox

        try:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = self.roi_bbox

            # Search in expanded region
            search_x = max(0, x - w // 2)
            search_y = max(0, y - h // 2)
            search_w = min(frame_gray.shape[1] - search_x, w * 2)
            search_h = min(frame_gray.shape[0] - search_y, h * 2)

            if search_w < w or search_h < h:
                return False, self.roi_bbox

            search_region = frame_gray[search_y:search_y+search_h, search_x:search_x+search_w]

            # Resize template if needed
            template = self.roi_template
            if template.shape[0] > search_h or template.shape[1] > search_w:
                scale = min(search_h / template.shape[0], search_w / template.shape[1])
                new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
                template = cv2.resize(template, new_size, interpolation=cv2.INTER_LINEAR)

            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # Higher threshold for template matching to avoid false positives
            if max_val > 1000:  # Stricter threshold
                top_left = max_loc
                new_x = search_x + top_left[0]
                new_y = search_y + top_left[1]
                new_w = template.shape[1]
                new_h = template.shape[0]

                # Validate movement isn't too large
                if self.last_valid_bbox is not None:
                    old_x, old_y, old_w, old_h = self.last_valid_bbox
                    center_dist = abs(new_x + new_w//2 - (old_x + old_w//2)) + abs(new_y + new_h//2 - (old_y + old_h//2))
                    if center_dist > max(old_w, old_h) * 3:
                        # Movement too large -> spurious match
                        return False, self.roi_bbox

                self.roi_bbox = (new_x, new_y, new_w, new_h)
                self.last_valid_bbox = self.roi_bbox
                self.consecutive_lost_frames = 0
                return True, self.roi_bbox
            return False, self.roi_bbox
        except:
            return False, self.roi_bbox

    def _create_csrt_tracker(self):
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()

        legacy = getattr(cv2, "legacy", None)
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        return None


class VideoLabel(QtWidgets.QLabel):
    roiSelected = QtCore.pyqtSignal(tuple)  # (x, y, w, h) in frame coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background-color:#111; border:1px solid #333;")

        self._select_enabled = False
        self._dragging = False
        self._p1 = QtCore.QPoint()
        self._p2 = QtCore.QPoint()

        # mapping info
        self._frame_w = 1
        self._frame_h = 1
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

    def enable_selection(self, enabled: bool):
        self._select_enabled = enabled
        self._dragging = False
        self._p1 = QtCore.QPoint()
        self._p2 = QtCore.QPoint()
        self.update()

    def set_frame_geometry_info(self, frame_w, frame_h, scale, offset_x, offset_y):
        self._frame_w = max(1, int(frame_w))
        self._frame_h = max(1, int(frame_h))
        self._scale = float(scale)
        self._offset_x = int(offset_x)
        self._offset_y = int(offset_y)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if not self._select_enabled:
            return super().mousePressEvent(event)
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self._p1 = event.pos()
            self._p2 = event.pos()
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._select_enabled and self._dragging:
            self._p2 = event.pos()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if not self._select_enabled:
            return super().mouseReleaseEvent(event)

        if event.button() == QtCore.Qt.LeftButton and self._dragging:
            self._dragging = False
            self._p2 = event.pos()
            self.update()

            rect = QtCore.QRect(self._p1, self._p2).normalized()
            if rect.width() < 5 or rect.height() < 5:
                return

            fx1, fy1 = self._label_to_frame(rect.left(), rect.top())
            fx2, fy2 = self._label_to_frame(rect.right(), rect.bottom())

            x1, y1 = min(fx1, fx2), min(fy1, fy2)
            x2, y2 = max(fx1, fx2), max(fy1, fy2)

            x1 = max(0, min(self._frame_w - 1, x1))
            y1 = max(0, min(self._frame_h - 1, y1))
            x2 = max(0, min(self._frame_w - 1, x2))
            y2 = max(0, min(self._frame_h - 1, y2))

            w = x2 - x1
            h = y2 - y1
            if w > 5 and h > 5:
                self.roiSelected.emit((int(x1), int(y1), int(w), int(h)))

    def _label_to_frame(self, lx, ly):
        x = (lx - self._offset_x) / self._scale
        y = (ly - self._offset_y) / self._scale
        return int(round(x)), int(round(y))

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        if self._select_enabled and (self._dragging or (self._p1 != self._p2)):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(0, 200, 255), 2, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 200, 255, 40)))
            painter.drawRect(QtCore.QRect(self._p1, self._p2).normalized())
            painter.end()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Tracking - MP4 loop (No Serial)")
        self.resize(1100, 700)

        # video
        self.cap = None
        self.video_path = ""
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)  # giảm jitter timer
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_frame)

        # tracking
        self.tracker = None
        self.tracking = False

        # tách frame gốc và frame vẽ
        self.raw_frame = None
        self.frame_w = 0
        self.frame_h = 0

        self.objX = 0.0
        self.objY = 0.0
        self.areaObj = 0.0

        self._prev_t = time.time()
        self._fps = 0.0

        self._resume_after_roi = False
        self._lost_reported = False
        self._consecutive_lost_for_roi = 0  # Consecutive lost frames before auto ROI

        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        root.addLayout(top, 1)

        self.video_label = VideoLabel()
        self.video_label.setFixedSize(800, 450)  # cố định kích thước để tránh giật
        self.video_label.roiSelected.connect(self.on_roi_selected)
        top.addWidget(self.video_label, 0)  # stretch = 0 để không resize

        side = QtWidgets.QVBoxLayout()
        side.setContentsMargins(5, 0, 0, 0)
        side.setSpacing(5)
        top.addLayout(side, 0)

        gb_video = QtWidgets.QGroupBox("Video (.mp4)")
        gb_video.setMaximumHeight(120)
        side.addWidget(gb_video)
        v = QtWidgets.QVBoxLayout(gb_video)
        v.setContentsMargins(5, 5, 5, 5)
        v.setSpacing(3)

        row = QtWidgets.QHBoxLayout()
        self.edt_path = QtWidgets.QLineEdit()
        self.edt_path.setPlaceholderText("Chọn file video .mp4 ...")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_video)
        row.addWidget(self.edt_path, 1)
        row.addWidget(btn_browse)
        v.addLayout(row)

        gb_ctrl = QtWidgets.QGroupBox("Điều khiển")
        gb_ctrl.setMaximumHeight(200)
        side.addWidget(gb_ctrl)
        g = QtWidgets.QGridLayout(gb_ctrl)
        g.setContentsMargins(5, 5, 5, 5)
        g.setSpacing(5)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_roi = QtWidgets.QPushButton("Chọn ROI (kéo trên video)")
        self.btn_exit = QtWidgets.QPushButton("Exit")

        self.btn_start.clicked.connect(self.start_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_reset.clicked.connect(self.reset_tracking)
        self.btn_roi.clicked.connect(self.toggle_roi_mode)
        self.btn_exit.clicked.connect(self.close)

        g.addWidget(self.btn_start, 0, 0)
        g.addWidget(self.btn_pause, 0, 1)
        g.addWidget(self.btn_reset, 1, 0)
        g.addWidget(self.btn_exit, 1, 1)
        g.addWidget(self.btn_roi, 2, 0, 1, 2)

        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setStyleSheet("color:#DDD;")
        side.addWidget(self.lbl_status)

        self.txt_log = QtWidgets.QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(220)
        side.addWidget(self.txt_log, 1)

        side.addStretch()

    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.txt_log.append(f"[{t}] {msg}")

    def set_status(self, msg):
        self.lbl_status.setText(f"Status: {msg}")

    # ---------- Video ----------
    def browse_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Chọn video", "", "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)"
        )
        if path:
            self.edt_path.setText(path)

    def start_video(self):
        path = self.edt_path.text().strip()
        if not path:
            self.log("Chưa chọn video. Hãy Browse và chọn file .mp4.")
            return

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.log(f"Không mở được video: {path}")
            self.set_status("Open video failed")
            return

        self.video_path = path

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 1:
            self.timer.setInterval(int(1000 / fps))

        self.reset_tracking(inform=False)
        self.timer.start()
        self.set_status("Playing (auto loop)")
        self.log("Video started. Bấm 'Chọn ROI' rồi kéo trên video để tracking.")

    def pause_video(self):
        if self.timer.isActive():
            self.timer.stop()
            self.set_status("Paused")
            self.log("Pause")
        else:
            if self.cap is not None:
                self.timer.start()
                self.set_status("Playing")
                self.log("Resume")

    def reset_tracking(self, checked=False, inform=True):
        self.setUpdatesEnabled(False)

        self.tracker = None
        self.tracking = False
        self.objX = self.objY = self.areaObj = 0.0
        self._lost_reported = False
        self._consecutive_lost_for_roi = 0  # Reset lost counter

        self.video_label.enable_selection(False)
        self.btn_roi.setText("Chọn ROI (kéo trên video)")

        # đè frame sạch ngay để xoá khung cũ
        self.refresh_view()

        self.setUpdatesEnabled(True)

        if inform:
            self.set_status("Reset tracking (select ROI again)")
            self.log("Reset tracking. Bấm 'Chọn ROI' để chọn lại đối tượng.")

    # ---------- ROI mode ----------
    def set_roi_mode(self, enabled: bool, auto: bool = False):
        if enabled:
            if getattr(self.video_label, "_select_enabled", False):
                return

            # pause để kéo ROI mượt
            self._resume_after_roi = self.timer.isActive()
            if self.timer.isActive():
                self.timer.stop()

            # Ensure tracking is fully stopped
            if self.tracking or self.tracker is not None:
                self.tracking = False
                self.tracker = None

            # block layout updates để tránh giật
            self.setUpdatesEnabled(False)

            self.video_label.enable_selection(True)
            self.btn_roi.setText("Đang chọn ROI... (bấm để tắt)")
            self.set_status("ROI mode ON (video paused) - kéo thả trên video")

            # luôn refresh frame sạch để tránh còn bbox cũ
            self.refresh_view()

            self.setUpdatesEnabled(True)
            self.update()

            if auto:
                self.log("Tracking lost -> AUTO bật ROI mode (video paused).")
            else:
                self.log("ROI mode ON: video tạm dừng để chọn ROI mượt.")

        else:
            if not getattr(self.video_label, "_select_enabled", False):
                return

            # block layout updates
            self.setUpdatesEnabled(False)

            self.video_label.enable_selection(False)
            self.btn_roi.setText("Chọn ROI (kéo trên video)")
            self.set_status("ROI mode OFF")

            self.setUpdatesEnabled(True)
            self.update()

            self.log("ROI mode OFF")

            # resume có delay nhỏ để tránh giật ngay sau thả chuột
            if self._resume_after_roi and self.cap is not None:
                QtCore.QTimer.singleShot(60, self.timer.start)
            self._resume_after_roi = False

    def toggle_roi_mode(self):
        if self.cap is None:
            self.log("Chưa chạy video. Bấm Start trước.")
            return
        enabled = not getattr(self.video_label, "_select_enabled", False)
        self.set_roi_mode(enabled, auto=False)

    def on_auto_roi_activation(self):
        """Safely activate ROI mode when tracking is lost"""
        # Double check that tracking is indeed off
        if self.tracking or self.tracker is not None:
            self.tracking = False
            self.tracker = None

        # Activate ROI selection mode
        self.set_roi_mode(True, auto=True)

    def on_roi_selected(self, bbox):
        if self.raw_frame is None:
            self.log("Chưa có frame để init tracker.")
            return

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            self.log("ROI không hợp lệ.")
            return

        try:
            # block updates khi init tracker để tránh giật
            self.setUpdatesEnabled(False)

            self.tracker = create_tracker()
            ok = self.tracker.init(self.raw_frame, (x, y, w, h))

            self.setUpdatesEnabled(True)

            if ok:
                self.tracking = True
                self._lost_reported = False

                self.video_label.enable_selection(False)
                self.btn_roi.setText("Chọn ROI (kéo trên video)")
                self.set_status(f"Tracking ON: ({x},{y},{w},{h})")

                # Log which tracker method was selected
                if hasattr(self.tracker, 'use_template') and self.tracker.use_template:
                    self.log(f"✓ Tracker initialized (Template Matching mode): ({x},{y},{w},{h})")
                elif hasattr(self.tracker, 'use_csrt') and self.tracker.use_csrt:
                    self.log(f"✓ Tracker initialized (CSRT mode): ({x},{y},{w},{h})")
                else:
                    self.log(f"✓ Tracker initialized (Feature-based mode): ({x},{y},{w},{h})")

                # reset FPS smoothing để tránh "giật" ngay sau init
                self._prev_t = time.time()
                self._fps = 0.0

                # resume có delay nhỏ để UI ổn định
                if self._resume_after_roi:
                    QtCore.QTimer.singleShot(60, self.timer.start)
                self._resume_after_roi = False
            else:
                detail = getattr(self.tracker, "last_error", "")
                if detail:
                    self.log(f"⚠ Init tracker FAILED: {detail}")
                else:
                    self.log("⚠ Init tracker FAILED. Thử chọn ROI khác.")
        except Exception as e:
            self.setUpdatesEnabled(True)
            self.log(f"Lỗi tracker: {e}")

    # ---------- Frame update ----------
    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.log("Loop video failed. Stop.")
                self.timer.stop()
                return

            self.reset_tracking(inform=False)
            self.log("Video loop về đầu. Tracker reset (hãy chọn ROI lại nếu cần).")

        self.raw_frame = frame
        self.frame_h, self.frame_w = frame.shape[:2]

        display = self.raw_frame.copy()

        if self.tracking and self.tracker is not None:
            ok, bbox = self.tracker.update(self.raw_frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                x = max(0, min(self.frame_w - 1, x))
                y = max(0, min(self.frame_h - 1, y))
                w = max(1, min(self.frame_w - x, w))
                h = max(1, min(self.frame_h - y, h))

                # Draw tracking bbox in GREEN (success)
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)

                self.objX = round(((cx - self.frame_w / 2) / (self.frame_w / 2)), 3)
                self.objY = round(((cy - self.frame_h / 2) / (self.frame_h / 2)), 3)
                self.areaObj = round(((w * h) / (self.frame_w * self.frame_h)) * 100, 3)

                cv2.putText(display, f"objX={self.objX} objY={self.objY} area={self.areaObj}%",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

                self._lost_reported = False
                self._consecutive_lost_for_roi = 0  # Reset lost counter on success

            else:
                # LOST: increment lost counter and show RED bbox
                self._consecutive_lost_for_roi += 1

                # Try to get last known bbox position (prefer last_valid_bbox over roi_bbox)
                bbox_to_draw = None
                if hasattr(self.tracker, 'last_valid_bbox') and self.tracker.last_valid_bbox is not None:
                    bbox_to_draw = self.tracker.last_valid_bbox
                elif hasattr(self.tracker, 'roi_bbox') and self.tracker.roi_bbox is not None:
                    bbox_to_draw = self.tracker.roi_bbox

                # Draw tracking bbox in RED (lost) if we have a position
                if bbox_to_draw is not None:
                    x, y, w, h = [int(v) for v in bbox_to_draw]
                    x = max(0, min(self.frame_w - 1, x))
                    y = max(0, min(self.frame_h - 1, y))
                    w = max(1, min(self.frame_w - x, w))
                    h = max(1, min(self.frame_h - y, h))

                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cx, cy = x + w // 2, y + h // 2
                    cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)

                cv2.putText(display, f"🔴 Tracking LOST ({self._consecutive_lost_for_roi} frame)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Only auto-enable ROI after sustained loss (5+ consecutive frames)
                if self._consecutive_lost_for_roi >= 5:
                    if not self._lost_reported:
                        self._lost_reported = True
                        self.log(f"Tracking lost for {self._consecutive_lost_for_roi} frames -> tự bật ROI mode.")
                        # Reset tracker to stop attempting failed updates
                        self.tracking = False
                        self.tracker = None
                        # Immediately refresh to clear old bbox from display
                        self.refresh_view()
                        # bật ROI mode với delay để ensure UI stable
                        QtCore.QTimer.singleShot(150, lambda: self.on_auto_roi_activation())

        # FPS
        now = time.time()
        dt = now - self._prev_t
        self._prev_t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        cv2.putText(display, f"FPS: {self._fps:.1f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        self.show_frame_on_label(display)

    def refresh_view(self):
        """Đè lại frame sạch ngay (để không còn bbox cũ trên QLabel)."""
        if self.raw_frame is None:
            return
        self.show_frame_on_label(self.raw_frame)

    # ---------- Optimized render ----------
    def show_frame_on_label(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        label_w = max(1, self.video_label.width())
        label_h = max(1, self.video_label.height())

        # keep aspect ratio (cố định 800x450)
        scale = min(label_w / w, label_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        offset_x = (label_w - new_w) // 2
        offset_y = (label_h - new_h) // 2

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        qimg = QtGui.QImage(resized.data, new_w, new_h, new_w * 3, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)

        # set pixmap trực tiếp không qua event queue -> response lập tức
        self.video_label.setPixmap(pm)
        self.video_label.set_frame_geometry_info(w, h, scale, offset_x, offset_y)

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except:
            pass
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())