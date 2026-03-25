import sys
import time
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

MIN_GOOD_MATCHES = 4       # tối thiểu cho homography
MIN_MEDIAN_MATCHES = 2     # tối thiểu cho fallback median-shift
LOWE_RATIO = 0.8           # ratio test (nới ra cho ROI nhỏ)
SEARCH_MARGIN = 2.5        # mở rộng vùng search quanh vị trí cuối
MAX_AREA_CHANGE = 3.0      # bbox mới không được > 3x hoặc < 1/3 bbox cũ
CONTEXT_PAD = 0.3           # mở rộng ROI thêm 30% mỗi chiều khi init (lấy context)
TEMPLATE_THRESH = 0.45      # ngưỡng NCC cho template matching fallback
ADAPT_INTERVAL = 5          # chỉ update model mỗi N frames thành công (tránh drift)


class SurfTracker:
    """SURF feature-based tracker + template matching fallback.
    API tương thích: init(frame, bbox) -> bool, update(frame) -> (ok, bbox)
    """

    def __init__(self, hessian_threshold=100):
        self.surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
        self.surf.setExtended(True)   # 128-dim descriptors
        self.bf = cv2.BFMatcher(cv2.NORM_L2)  # BF ổn định hơn FLANN cho ít features

        self.ref_kp = None
        self.ref_des = None
        self.ref_pts = None       # 4 corners của ROI gốc (float32)
        self.last_bbox = None     # (x, y, w, h) vị trí cuối
        self.init_area = 0
        self.lost_count = 0
        self.success_count = 0    # đếm frames tracking OK liên tiếp

        # template matching
        self.ref_template = None  # grayscale template từ init
        self.ref_bbox_wh = (0, 0)  # (w, h) gốc

        # context-expanded reference (cho SURF matching chính xác hơn)
        self.ref_kp_ctx = None
        self.ref_des_ctx = None

    def init(self, frame, bbox):
        """Khởi tạo tracker với frame và bbox = (x, y, w, h)."""
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Lưu template gốc cho NCC fallback ---
        roi_tmpl = gray[y:y+h, x:x+w].copy()
        self.ref_template = roi_tmpl
        self.ref_bbox_wh = (w, h)

        # --- Mở rộng ROI lấy context cho SURF (nhiều features hơn) ---
        pad_x = int(w * CONTEXT_PAD)
        pad_y = int(h * CONTEXT_PAD)
        cx1 = max(0, x - pad_x)
        cy1 = max(0, y - pad_y)
        cx2 = min(fw, x + w + pad_x)
        cy2 = min(fh, y + h + pad_y)
        roi_ctx = gray[cy1:cy2, cx1:cx2]

        kp_ctx, des_ctx = self.surf.detectAndCompute(roi_ctx, None)
        if des_ctx is not None and len(kp_ctx) >= MIN_MEDIAN_MATCHES:
            for k in kp_ctx:
                k.pt = (k.pt[0] + cx1, k.pt[1] + cy1)
            self.ref_kp_ctx = kp_ctx
            self.ref_des_ctx = des_ctx.astype(np.float32)

        # --- SURF trên ROI chính xác ---
        roi = gray[y:y+h, x:x+w]
        kp, des = self.surf.detectAndCompute(roi, None)

        # Nếu ROI chính ko đủ features, dùng context features
        if des is None or len(kp) < MIN_MEDIAN_MATCHES:
            if self.ref_des_ctx is not None:
                self.ref_kp = list(self.ref_kp_ctx)
                self.ref_des = self.ref_des_ctx.copy()
            else:
                # Cả 2 đều fail nhưng vẫn có template matching
                self.ref_kp = None
                self.ref_des = None
        else:
            for k in kp:
                k.pt = (k.pt[0] + x, k.pt[1] + y)
            self.ref_kp = kp
            self.ref_des = des.astype(np.float32)

        # Cần ít nhất template HOẶC SURF features
        if self.ref_des is None and self.ref_template is None:
            return False

        self.ref_pts = np.float32([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ]).reshape(-1, 1, 2)
        self.last_bbox = (x, y, w, h)
        self.init_area = w * h
        self.lost_count = 0
        self.success_count = 0
        return True

    def _validate_bbox(self, x, y, w, h, fw, fh):
        if w < 5 or h < 5 or w > fw or h > fh:
            return False
        area = w * h
        if self.init_area > 0:
            ratio = area / self.init_area
            if ratio > MAX_AREA_CHANGE or ratio < (1.0 / MAX_AREA_CHANGE):
                return False
        # aspect ratio check: prevent degenerate bbox from homography
        if self.ref_bbox_wh[0] > 0 and self.ref_bbox_wh[1] > 0:
            orig_ar = self.ref_bbox_wh[0] / self.ref_bbox_wh[1]
            new_ar = w / h
            ar_ratio = new_ar / orig_ar if orig_ar > 0 else 1.0
            if ar_ratio > 2.0 or ar_ratio < 0.5:
                return False
        return True

    def _validate_homography(self, H):
        if H is None:
            return False
        det = np.linalg.det(H[:2, :2])
        if det < 0.1 or det > 10.0:
            return False
        return True

    def _median_shift_bbox(self, src_pts, dst_pts):
        dx = np.median(dst_pts[:, 0, 0] - src_pts[:, 0, 0])
        dy = np.median(dst_pts[:, 0, 1] - src_pts[:, 0, 1])
        lx, ly, lw, lh = self.last_bbox
        nx = int(round(lx + dx))
        ny = int(round(ly + dy))
        return nx, ny, lw, lh

    def _template_match(self, gray, fw, fh):
        """Template matching (NCC) trong search region. Return (ok, (x,y,w,h))."""
        if self.ref_template is None:
            return False, (0, 0, 0, 0)

        tw, th = self.ref_bbox_wh
        lx, ly, lw, lh = self.last_bbox

        # search region
        margin = max(tw, th) * 3
        sx = max(0, int(lx - margin))
        sy = max(0, int(ly - margin))
        ex = min(fw, int(lx + lw + margin))
        ey = min(fh, int(ly + lh + margin))
        search_roi = gray[sy:ey, sx:ex]

        if search_roi.shape[0] < th or search_roi.shape[1] < tw:
            return False, (0, 0, 0, 0)

        # multi-scale template matching
        best_val = -1
        best_loc = None
        best_scale = 1.0

        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            sw = max(5, int(tw * scale))
            sh = max(5, int(th * scale))
            if sw >= search_roi.shape[1] or sh >= search_roi.shape[0]:
                continue
            scaled_tmpl = cv2.resize(self.ref_template, (sw, sh))
            res = cv2.matchTemplate(search_roi, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale

        if best_val < TEMPLATE_THRESH or best_loc is None:
            return False, (0, 0, 0, 0)

        rx = sx + best_loc[0]
        ry = sy + best_loc[1]
        rw = int(tw * best_scale)
        rh = int(th * best_scale)

        if self._validate_bbox(rx, ry, rw, rh, fw, fh):
            return True, (rx, ry, rw, rh)
        return False, (0, 0, 0, 0)

    def _update_reference(self, frame, bbox):
        """Cập nhật template và SURF reference (chỉ khi confidence cao)."""
        x, y, w, h = [int(v) for v in bbox]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]
        x = max(0, min(fw - w, x))
        y = max(0, min(fh - h, y))
        if w < 5 or h < 5:
            return

        # Update template (nhẹ blend)
        new_tmpl = gray[y:y+h, x:x+w]
        if new_tmpl.shape == self.ref_template.shape:
            alpha = 0.15
            self.ref_template = cv2.addWeighted(
                self.ref_template, 1 - alpha, new_tmpl, alpha, 0)

        # Update SURF features
        if self.ref_des is None:
            return

        roi = gray[y:y+h, x:x+w]
        kp_new, des_new = self.surf.detectAndCompute(roi, None)
        if des_new is None or len(kp_new) < MIN_MEDIAN_MATCHES:
            return

        for k in kp_new:
            k.pt = (k.pt[0] + x, k.pt[1] + y)
        des_new = des_new.astype(np.float32)

        # Merge: 80% cũ + 20% mới
        n_old = max(1, int(len(self.ref_kp) * 0.8))
        n_new = max(1, int(len(kp_new) * 0.2))

        old_idx = sorted(range(len(self.ref_kp)),
                         key=lambda i: self.ref_kp[i].response, reverse=True)[:n_old]
        new_idx = sorted(range(len(kp_new)),
                         key=lambda i: kp_new[i].response, reverse=True)[:n_new]

        self.ref_kp = [self.ref_kp[i] for i in old_idx] + [kp_new[i] for i in new_idx]
        self.ref_des = np.vstack([self.ref_des[old_idx], des_new[new_idx]])

        self.ref_pts = np.float32([
            [x, y], [x + w, y], [x + w, y + h], [x, y + h]
        ]).reshape(-1, 1, 2)

    def update(self, frame):
        """Tìm object trong frame mới. Return (ok, (x, y, w, h))."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]

        surf_ok = False
        sx, sy, sw, sh = 0, 0, 0, 0

        # ===== Bước 1: SURF feature matching =====
        if self.ref_des is not None and len(self.ref_des) >= MIN_MEDIAN_MATCHES:
            margin_mult = SEARCH_MARGIN + self.lost_count * 0.5
            margin_mult = min(margin_mult, 5.0)

            mask = None
            if self.last_bbox is not None:
                lx, ly, lw, lh = self.last_bbox
                margin_w = int(lw * margin_mult)
                margin_h = int(lh * margin_mult)
                sx0 = max(0, lx - margin_w)
                sy0 = max(0, ly - margin_h)
                ex0 = min(fw, lx + lw + margin_w)
                ey0 = min(fh, ly + lh + margin_h)
                mask = np.zeros((fh, fw), dtype=np.uint8)
                mask[sy0:ey0, sx0:ex0] = 255

            kp, des = self.surf.detectAndCompute(gray, mask)
            if des is not None and len(kp) >= MIN_MEDIAN_MATCHES:
                des = des.astype(np.float32)
                try:
                    matches = self.bf.knnMatch(self.ref_des, des, k=2)
                    good = []
                    for pair in matches:
                        if len(pair) == 2:
                            m, n = pair
                            if m.distance < LOWE_RATIO * n.distance:
                                good.append(m)

                    if len(good) >= MIN_MEDIAN_MATCHES:
                        src_pts = np.float32(
                            [self.ref_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32(
                            [kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                        # Thử homography
                        if len(good) >= MIN_GOOD_MATCHES:
                            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if self._validate_homography(H):
                                dst_corners = cv2.perspectiveTransform(self.ref_pts, H)
                                rect = cv2.boundingRect(dst_corners)
                                bx, by, bw, bh = rect
                                if self._validate_bbox(bx, by, bw, bh, fw, fh):
                                    sx, sy, sw, sh = bx, by, bw, bh
                                    surf_ok = True

                        # Fallback: median-shift
                        if not surf_ok:
                            mx, my, mw, mh = self._median_shift_bbox(src_pts, dst_pts)
                            if self._validate_bbox(mx, my, mw, mh, fw, fh):
                                sx, sy, sw, sh = mx, my, mw, mh
                                surf_ok = True
                except cv2.error:
                    pass

        # ===== Bước 2: Template matching fallback/verification =====
        tmpl_ok = False
        tx, ty, tw, th = 0, 0, 0, 0

        if self.ref_template is not None:
            tmpl_ok, (tx, ty, tw, th) = self._template_match(gray, fw, fh)

        # ===== Bước 3: Chọn kết quả tốt nhất =====
        if surf_ok and tmpl_ok:
            # Cả 2 OK → dùng SURF (chính xác hơn) nhưng kiểm tra consistency
            dist = np.sqrt((sx - tx)**2 + (sy - ty)**2)
            orig_w, orig_h = self.ref_bbox_wh
            max_dist = max(orig_w, orig_h) * 1.5
            if dist < max_dist:
                x, y, w, h = sx, sy, sw, sh  # consistent → dùng SURF
            else:
                x, y, w, h = tx, ty, tw, th  # inconsistent → tin template hơn
        elif surf_ok:
            x, y, w, h = sx, sy, sw, sh
        elif tmpl_ok:
            x, y, w, h = tx, ty, tw, th
        else:
            self.lost_count += 1
            return False, (0, 0, 0, 0)

        x = max(0, min(fw - w, x))
        y = max(0, min(fh - h, y))

        self.last_bbox = (x, y, w, h)
        self.lost_count = 0
        self.success_count += 1

        # Chỉ update reference mỗi ADAPT_INTERVAL frames (tránh drift)
        if self.success_count % ADAPT_INTERVAL == 0:
            self._update_reference(frame, (x, y, w, h))

        return True, (x, y, w, h)


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
        self.setWindowTitle("SURF Tracking - MP4 loop")
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

        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top, 1)

        self.video_label = VideoLabel()
        self.video_label.setMinimumSize(800, 450)
        self.video_label.roiSelected.connect(self.on_roi_selected)
        top.addWidget(self.video_label, 3)

        side = QtWidgets.QVBoxLayout()
        top.addLayout(side, 1)

        gb_video = QtWidgets.QGroupBox("Video (.mp4)")
        side.addWidget(gb_video)
        v = QtWidgets.QVBoxLayout(gb_video)

        row = QtWidgets.QHBoxLayout()
        self.edt_path = QtWidgets.QLineEdit()
        self.edt_path.setPlaceholderText("Chọn file video .mp4 ...")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_video)
        row.addWidget(self.edt_path, 1)
        row.addWidget(btn_browse)
        v.addLayout(row)

        gb_ctrl = QtWidgets.QGroupBox("Điều khiển")
        side.addWidget(gb_ctrl)
        g = QtWidgets.QGridLayout(gb_ctrl)

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
        self.tracker = None
        self.tracking = False
        self.objX = self.objY = self.areaObj = 0.0
        self._lost_reported = False

        self.video_label.enable_selection(False)
        self.btn_roi.setText("Chọn ROI (kéo trên video)")

        # đè frame sạch ngay để xoá khung cũ
        self.refresh_view()

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

            self.video_label.enable_selection(True)
            self.btn_roi.setText("Đang chọn ROI... (bấm để tắt)")
            self.set_status("ROI mode ON (video paused) - kéo thả trên video")

            if auto:
                self.log("Tracking lost -> AUTO bật ROI mode (video paused).")
            else:
                self.log("ROI mode ON: video tạm dừng để chọn ROI mượt.")

            # luôn refresh frame sạch để tránh còn bbox cũ
            self.refresh_view()

        else:
            if not getattr(self.video_label, "_select_enabled", False):
                return

            self.video_label.enable_selection(False)
            self.btn_roi.setText("Chọn ROI (kéo trên video)")
            self.set_status("ROI mode OFF")
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

    def on_roi_selected(self, bbox):
        if self.raw_frame is None:
            self.log("Chưa có frame để init tracker.")
            return

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            self.log("ROI không hợp lệ.")
            return

        try:
            self.tracker = SurfTracker()
            ok = self.tracker.init(self.raw_frame, (x, y, w, h))
            if ok:
                self.tracking = True
                self._lost_reported = False

                self.video_label.enable_selection(False)
                self.btn_roi.setText("Chọn ROI (kéo trên video)")
                self.set_status(f"Tracking ON: ({x},{y},{w},{h})")
                self.log(f"Init tracker OK: ({x},{y},{w},{h})")

                # reset FPS smoothing để tránh “giật” ngay sau init
                self._prev_t = time.time()
                self._fps = 0.0

                # resume có delay nhỏ để UI ổn định
                if self._resume_after_roi:
                    QtCore.QTimer.singleShot(60, self.timer.start)
                self._resume_after_roi = False
            else:
                self.log("Init tracker FAILED. Thử chọn ROI khác.")
        except Exception as e:
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

                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = x + w // 2, y + h // 2
                cv2.circle(display, (cx, cy), 4, (0, 255, 0), -1)

                self.objX = round(((cx - self.frame_w / 2) / (self.frame_w / 2)), 3)
                self.objY = round(((cy - self.frame_h / 2) / (self.frame_h / 2)), 3)
                self.areaObj = round(((w * h) / (self.frame_w * self.frame_h)) * 100, 3)

                cv2.putText(display, f"objX={self.objX} objY={self.objY} area={self.areaObj}%",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

                self._lost_reported = False

            else:
                # LOST: xóa tracker để chắc chắn không giữ bbox
                self.tracking = False
                self.tracker = None

                cv2.putText(display, "Tracking LOST - ROI auto ON",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not self._lost_reported:
                    self._lost_reported = True
                    self.log("Tracking lost -> tự bật ROI mode.")
                    # bật ROI mode và refresh frame sạch để không còn bbox cũ
                    QtCore.QTimer.singleShot(0, lambda: self.set_roi_mode(True, auto=True))

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

        # keep aspect ratio
        scale = min(label_w / w, label_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        offset_x = (label_w - new_w) // 2
        offset_y = (label_h - new_h) // 2

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        qimg = QtGui.QImage(resized.data, new_w, new_h, new_w * 3, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg)

        # ✅ Không tạo canvas to bằng label nữa -> giảm giật
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