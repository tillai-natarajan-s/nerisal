#!/usr/bin/env python3
"""
NERISAL Live - real-time edge crowd-pressure monitor (Team Iconic)

Camera -> fast ExecuTorch detector every frame ("watch" tier)
       -> escalates to the accurate tiled detector when density rises or the
          saturation guard fires ("escalated" tier)
       -> density, Fruin grade A-F, fail-safe, in/out line counting,
          time-to-critical prediction
       -> browser dashboard served by the Pi itself (no internet, no extra packages)

Runs on Raspberry Pi 4 or 5. Needs only: opencv, numpy, executorch (+ nerisal.py
in the same folder). Pi Camera needs picamera2 (apt), see --source.

Examples
  python nerisal_live.py --source rpicam                      # Pi Camera, no Python camera lib needed
  python nerisal_live.py --source picam                       # Pi Camera via picamera2
  python nerisal_live.py --source 0                           # USB webcam
  python nerisal_live.py --source clip.mp4                    # organiser video, looped in real time
  python nerisal_live.py --source images/ --hold 4            # image folder slideshow
  python nerisal_live.py --source 0 --area 25 --line 0.6      # 25 m^2 floor, counting line at 60% height
  python nerisal_live.py --source clip.mp4 --mock             # no ExecuTorch (laptop test, OpenCV HOG)
Then open  http://<pi-ip>:8000  on the laptop.
"""
import argparse, collections, csv, glob, json, os, socket, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# ---------------------------------------------------------------- grading ----
FRUIN = [  # walkway Level of Service, persons per m^2 (upper bounds)
    (0.31, "A", "Free flow"),
    (0.43, "B", "Minor conflicts"),
    (0.72, "C", "Restricted speed - watch the trend"),
    (1.08, "D", "Crowded - monitor closely, prepare staff"),
    (2.17, "E", "Severe - hold entry, open an extra exit"),
    (float("inf"), "F", "Critical - crush risk, intervene now"),
]
E_LIMIT = 1.08
GRADE_HEX = {"A": "#2e9e5b", "B": "#5aa83c", "C": "#d9a400", "D": "#e07a10",
             "E": "#d64528", "E*": "#c0182b", "F": "#9b0d1f", "-": "#6b7280"}


def fruin(d):
    for lim, g, act in FRUIN:
        if d <= lim:
            return g, act


def texture(img):
    """Edge density (0..1). Packed crowds are uniformly busy."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(cv2.resize(img, (640, max(1, int(640 * h / w)))), cv2.COLOR_BGR2GRAY)
    return float(cv2.Canny(g, 80, 160).mean() / 255)


# ---------------------------------------------------------------- sources ----
class Source:
    """Background grabber that always holds the newest frame."""

    def __init__(self, spec, width, height, hold):
        self.spec, self.w, self.h, self.hold = spec, width, height, hold
        self.frame, self.seq, self.lock, self.alive = None, 0, threading.Lock(), True
        self.desc = spec
        self._open()
        threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        s = self.spec
        self.kind = None
        if s == "picam":
            from picamera2 import Picamera2          # apt: python3-picamera2
            self.cam = Picamera2()
            cfg = self.cam.create_video_configuration(main={"size": (self.w, self.h), "format": "RGB888"})
            self.cam.configure(cfg)
            self.cam.start()
            self.kind, self.desc = "picam", f"Pi Camera {self.w}x{self.h}"
        elif s == "rpicam":
            # No Python camera library needed: stream MJPEG from the rpicam-vid CLI.
            import subprocess
            exe = "rpicam-vid" if subprocess.call(["which", "rpicam-vid"], stdout=subprocess.DEVNULL) == 0 \
                else "libcamera-vid"
            self.proc = subprocess.Popen([exe, "-t", "0", "-n", "--codec", "mjpeg", "--width", str(self.w),
                                          "--height", str(self.h), "--framerate", "15", "-o", "-"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
            self.buf = b""
            self.kind, self.desc = "rpicam", f"Pi Camera via {exe} {self.w}x{self.h}"
        elif s.isdigit():
            self.cap = cv2.VideoCapture(int(s))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open camera {s}. Try another index, or --source picam.")
            self.kind, self.desc = "usb", f"USB camera {s}"
        elif os.path.isdir(s):
            exts = ("jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG")
            self.files = sorted(f for e in exts for f in glob.glob(os.path.join(s, f"*.{e}")))
            if not self.files:
                raise SystemExit(f"No images in {s}")
            self.kind, self.desc = "images", f"{len(self.files)} images from {s}"
        else:
            self.cap = cv2.VideoCapture(s)
            if not self.cap.isOpened():
                raise SystemExit(f"Cannot open video {s}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.kind, self.desc = "video", f"video {os.path.basename(s)} (looped)"

    def _put(self, f):
        if f is None:
            return
        with self.lock:
            self.frame, self.seq = f, self.seq + 1

    def _loop(self):
        i = 0
        while self.alive:
            if self.kind == "picam":
                self._put(self.cam.capture_array())
            elif self.kind == "rpicam":
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    time.sleep(0.05)
                    continue
                self.buf += chunk
                while True:
                    a = self.buf.find(b"\xff\xd8")
                    b = self.buf.find(b"\xff\xd9", a + 2)
                    if a < 0 or b < 0:
                        break
                    jpg, self.buf = self.buf[a:b + 2], self.buf[b + 2:]
                    self._put(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
                if len(self.buf) > 4_000_000:
                    self.buf = b""
            elif self.kind == "usb":
                ok, f = self.cap.read()
                self._put(f if ok else None)
                if not ok:
                    time.sleep(0.05)
            elif self.kind == "images":
                f = cv2.imread(self.files[i % len(self.files)])
                self._put(f)
                i += 1
                time.sleep(self.hold)
            else:  # video, paced to real time, looped
                ok, f = self.cap.read()
                if not ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._put(f)
                time.sleep(1.0 / self.fps)

    def latest(self):
        with self.lock:
            return self.frame, self.seq


# -------------------------------------------------------------- detectors ----
class ETDetector:
    """ExecuTorch YOLO via nerisal.py (same folder)."""

    def __init__(self, model, imgsz, conf, target, tiles, grid):
        from nerisal import Detector
        self.d = Detector(model, imgsz, conf, target)
        self.tiles, self.grid = tiles, grid
        self.names = self.d.names
        self.label = f"{os.path.basename(model)} @{imgsz}" + (f" tiled {grid}x{grid}" if tiles else "")

    def __call__(self, img):
        b, s, c, ms = self.d.detect(img, tiles=self.tiles, grid=self.grid)
        return b.numpy(), s.numpy(), c.numpy().astype(int), ms


class HOGDetector:
    """Stand-in for testing without ExecuTorch (OpenCV's classic people detector)."""

    def __init__(self, scale=1.0, label="mock HOG"):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.scale, self.names, self.label = scale, {0: "person"}, label

    def __call__(self, img):
        t = time.perf_counter()
        small = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        rects, w = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        ms = (time.perf_counter() - t) * 1000
        if len(rects) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), ms
        r = np.array(rects, float) / self.scale
        b = np.c_[r[:, 0], r[:, 1], r[:, 0] + r[:, 2], r[:, 1] + r[:, 3]]
        return b, np.array(w).ravel(), np.zeros(len(b), int), ms


# ---------------------------------------------------------------- tracking ---
class LineCounter:
    """Greedy centroid tracker + virtual counting line (horizontal) with hysteresis.
    A crossing counts only when a track moves from clearly above the line
    (beyond a dead band) to clearly below it, or the reverse, so detector jitter
    near the line is not counted."""

    def __init__(self, line_frac, max_jump=0.08, max_age=8, band=0.04, min_hits=3):
        self.line_frac, self.max_jump, self.max_age = line_frac, max_jump, max_age
        self.band, self.min_hits = band, min_hits
        self.tracks, self.next_id = {}, 1        # id -> dict(x, y, age, side, hits)
        self.down = self.up = 0

    def _side(self, y, ly, m):
        return -1 if y < ly - m else (1 if y > ly + m else 0)

    def update(self, boxes, shape):
        if self.line_frac <= 0:
            return
        h, w = shape[:2]
        ly, m, lim = self.line_frac * h, self.band * h, self.max_jump * max(h, w)
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        pairs = sorted((np.hypot(cx - t["x"], cy - t["y"]), tid, j)
                       for tid, t in self.tracks.items() for j, (cx, cy) in enumerate(cents))
        used_t, used_c = set(), set()
        for d, tid, j in pairs:                     # globally nearest pairs first
            if d > lim or tid in used_t or j in used_c:
                continue
            used_t.add(tid); used_c.add(j)
            t = self.tracks[tid]
            t["x"], t["y"], t["age"], t["hits"] = cents[j][0], cents[j][1], 0, t["hits"] + 1
            s = self._side(t["y"], ly, m)
            if s != 0:
                if t["side"] == -1 and s == 1 and t["hits"] >= self.min_hits:
                    self.down += 1
                elif t["side"] == 1 and s == -1 and t["hits"] >= self.min_hits:
                    self.up += 1
                t["side"] = s
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        for j, (cx, cy) in enumerate(cents):
            if j not in used_c:
                self.tracks[self.next_id] = {"x": cx, "y": cy, "age": 0, "hits": 1,
                                             "side": self._side(cy, ly, m)}
                self.next_id += 1


# ------------------------------------------------------------------ engine ---
class Engine:
    def __init__(self, a, src, fast, acc):
        self.a, self.src, self.fast, self.acc = a, src, fast, acc
        self.lock = threading.Lock()
        self.jpg, self.jseq = b"", 0
        self.counter = LineCounter(a.line)
        self.hist = collections.deque(maxlen=600)       # (t, count, density)
        self.events = collections.deque(maxlen=30)
        self.ema = None
        self.acc_res, self.acc_t, self.acc_ms = None, 0.0, 0.0
        self.loop_t = collections.deque(maxlen=30)
        self.state = {}
        self.prev_grade, self.prev_sat = None, False
        self.t0 = time.time()
        self.last_seq = -1
        os.makedirs(a.logdir, exist_ok=True)
        self.logf = open(os.path.join(a.logdir, time.strftime("session_%Y%m%d_%H%M%S.csv")), "w", newline="")
        self.log = csv.writer(self.logf)
        self.log.writerow(["time", "count", "density", "grade", "saturated", "mode", "fast_ms", "acc_ms",
                           "crossed_down", "crossed_up", "ttc_s"])
        self.last_log = 0.0

    def event(self, msg, level="info"):
        self.events.appendleft({"t": time.strftime("%H:%M:%S"), "msg": msg, "level": level})

    def ttc(self, now, d):
        """Seconds until density reaches Grade E at the current trend, or None."""
        pts = [(t, dd) for t, _, dd in self.hist if now - t <= self.a.trend]
        if len(pts) < 8 or pts[-1][0] - pts[0][0] < 5 or d >= E_LIMIT:
            return None
        t = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        slope = np.polyfit(t - t[0], y, 1)[0]
        if slope <= 1e-4:
            return None
        s = (E_LIMIT - d) / slope
        return s if s < 1800 else None

    def step(self):
        frame, seq = self.src.latest()
        if frame is None or seq == self.last_seq:     # wait for a new frame
            time.sleep(0.005)
            return
        self.last_seq = seq
        a, now = self.a, time.time()
        boxes, scores, cls, fms = self.fast(frame)
        names = self.fast.names
        self.counter.update(boxes, frame.shape)
        tex = texture(frame)
        n_fast = len(boxes)
        dens_fast = n_fast / a.area if a.area else 0.0

        # escalate to the accurate tier when it matters
        mode, shown_boxes, n = "watch", boxes, n_fast
        fresh = self.acc_res is not None and now - self.acc_t < a.hold_acc
        busy = tex >= a.tex_thr
        shown_cls, ran_acc = cls, False
        if self.acc and (busy or dens_fast >= a.esc_density) and now - self.acc_t >= a.esc_every:
            b2, s2, c2, ams = self.acc(frame)
            self.acc_res, self.acc_t, self.acc_ms, fresh, ran_acc = (b2, c2, len(b2)), now, ams, True, True
            self.event(f"Escalated: accurate pass found {len(b2)} (watch tier saw {n_fast})")
        if fresh and self.acc_res[2] >= n_fast:
            mode, n = "escalated", self.acc_res[2]
            if ran_acc:   # draw accurate boxes only on the frame they belong to
                shown_boxes, shown_cls = self.acc_res[0], self.acc_res[1]

        self.ema = n if self.ema is None else self.ema + a.smooth * (n - self.ema)
        cnt = int(round(self.ema))
        density = cnt / a.area if a.area else 0.0
        sat = a.target in ("person", "head") and busy and n < a.sat_count
        if a.area:
            grade, action = fruin(density)
        else:
            grade, action = "-", "Set --area to enable grading"
        if sat:
            grade = "E*" if grade in ("-", "A", "B", "C", "D") else grade
            action = "SATURATED - too dense to count. Treat as Grade E or worse."
        self.hist.append((now, cnt, density))
        ttc = self.ttc(now, density) if a.area and not sat else None

        if grade != self.prev_grade and self.prev_grade is not None:
            lvl = "alert" if grade in ("E", "E*", "F") else ("warn" if grade in ("C", "D") else "info")
            self.event(f"Grade {self.prev_grade} -> {grade}", lvl)
        if sat and not self.prev_sat:
            self.event("Saturation guard fired: crowd too dense to count", "alert")
        self.prev_grade, self.prev_sat = grade, sat

        img = self.annotate(frame.copy(), shown_boxes, shown_cls, names,
                            cnt, sat, grade, mode)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])

        self.loop_t.append(now)
        fps = (len(self.loop_t) - 1) / (self.loop_t[-1] - self.loop_t[0]) if len(self.loop_t) > 1 else 0.0
        st = {
            "count": cnt, "raw": n, "density": round(density, 3), "grade": grade, "action": action,
            "color": GRADE_HEX.get(grade, "#6b7280"), "saturated": sat, "mode": mode,
            "fast_ms": round(fms, 1), "acc_ms": round(self.acc_ms, 1), "fps": round(fps, 2),
            "down": self.counter.down, "up": self.counter.up, "line": a.line > 0,
            "ttc": None if ttc is None else int(ttc), "texture": round(tex, 3), "area": a.area,
            "source": self.src.desc, "fast_label": self.fast.label,
            "acc_label": self.acc.label if self.acc else "off",
            "uptime": int(now - self.t0),
            "hist": [[round(t - now, 1), c, round(d, 3)] for t, c, d in list(self.hist)[-240:]],
            "events": list(self.events)[:12],
        }
        with self.lock:
            if ok:
                self.jpg, self.jseq = jpg.tobytes(), self.jseq + 1
            self.state = st
        if now - self.last_log >= 1.0:
            self.last_log = now
            self.log.writerow([time.strftime("%H:%M:%S"), cnt, f"{density:.3f}", grade, int(sat), mode,
                               f"{fms:.1f}", f"{self.acc_ms:.1f}", self.counter.down, self.counter.up,
                               "" if ttc is None else int(ttc)])
            self.logf.flush()

    def annotate(self, img, boxes, cls, names, cnt, sat, grade, mode):
        h, w = img.shape[:2]
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in b]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            name = names.get(int(cls[i]), "") if cls is not None else "person"
            is_person = name in ("person", "head")
            if not self.a.no_blur and is_person:
                hy = y2 if name == "head" else y1 + max(1, (y2 - y1) // 5)   # head box: blur all of it
                roi = img[y1:hy, x1:x2]
                if roi.size:
                    img[y1:hy, x1:x2] = cv2.GaussianBlur(roi, (31, 31), 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60) if mode == "watch" else (255, 200, 40), 2)
        if self.a.line > 0:
            ly = int(self.a.line * h)
            cv2.line(img, (0, ly), (w, ly), (255, 0, 255), 2)
        bar = (40, 24, 200) if sat else (20, 20, 20)
        cv2.rectangle(img, (0, 0), (w, 34), bar, -1)
        txt = f"{'>=' if sat else ''}{cnt} people | LOS {grade} | {mode}"
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def run(self):
        while True:
            try:
                self.step()
            except Exception as e:                        # keep the demo alive
                self.event(f"error: {e}", "alert")
                print("engine error:", e)
                time.sleep(0.5)

    def snapshot(self):
        with self.lock:
            jpg, st = self.jpg, dict(self.state)
        os.makedirs(self.a.snapdir, exist_ok=True)
        name = time.strftime("snap_%H%M%S.jpg")
        with open(os.path.join(self.a.snapdir, name), "wb") as f:
            f.write(jpg)
        self.event(f"Snapshot saved: {name} ({st.get('count')} people, grade {st.get('grade')})")
        return name


# --------------------------------------------------------------- web server --
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


def make_handler(engine):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                with open(PAGE, "rb") as f:
                    self._send(200, "text/html; charset=utf-8", f.read())
            elif p == "/stats":
                with engine.lock:
                    body = json.dumps(engine.state).encode()
                self._send(200, "application/json", body)
            elif p == "/snapshot":
                self._send(200, "application/json", json.dumps({"saved": engine.snapshot()}).encode())
            elif p == "/frame.jpg":
                with engine.lock:
                    jpg = engine.jpg
                self._send(200, "image/jpeg", jpg)
            elif p == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = -1
                try:
                    while True:
                        with engine.lock:
                            jpg, seq = engine.jpg, engine.jseq
                        if seq != last and jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            last = seq
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self._send(404, "text/plain", b"not found")
    return H


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description="NERISAL live crowd-pressure monitor")
    ap.add_argument("--source", default="0", help="rpicam | picam | camera index (0) | video file | image folder")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hold", type=float, default=4.0, help="seconds per image in folder mode")
    ap.add_argument("--target", choices=["person", "head", "vehicle"], default="person",
                    help="head = our fine-tuned head models (use --conf 0.15)")
    ap.add_argument("--fast-model", default="yolov8n_320.pte")
    ap.add_argument("--fast-imgsz", type=int, default=320)
    ap.add_argument("--acc-model", default="yolov8n.pte", help="'' to disable escalation")
    ap.add_argument("--acc-imgsz", type=int, default=640)
    ap.add_argument("--grid", type=int, default=3, help="tiles per side in the accurate tier")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--area", type=float, default=0, help="visible floor area in m^2 (enables grades)")
    ap.add_argument("--line", type=float, default=0.0, help="counting line at this fraction of height, 0=off")
    ap.add_argument("--esc-density", type=float, default=0.43, help="escalate when watch density reaches this")
    ap.add_argument("--esc-every", type=float, default=3.0, help="min seconds between accurate passes")
    ap.add_argument("--hold-acc", type=float, default=6.0, help="seconds an accurate result stays valid")
    ap.add_argument("--smooth", type=float, default=0.35, help="count smoothing (0..1, 1=no smoothing)")
    ap.add_argument("--tex-thr", type=float, default=0.22)
    ap.add_argument("--sat-count", type=int, default=15)
    ap.add_argument("--trend", type=float, default=60, help="seconds of history for time-to-critical")
    ap.add_argument("--no-blur", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--logdir", default="live_logs")
    ap.add_argument("--snapdir", default="live_snaps")
    ap.add_argument("--mock", action="store_true", help="test without ExecuTorch (OpenCV HOG)")
    a = ap.parse_args()

    src = Source(a.source, a.width, a.height, a.hold)
    if a.mock:
        fast = HOGDetector(0.6, "mock HOG (fast)")
        acc = HOGDetector(1.0, "mock HOG (accurate)") if a.acc_model else None
    else:
        fast = ETDetector(a.fast_model, a.fast_imgsz, a.conf, a.target, False, 1)
        acc = ETDetector(a.acc_model, a.acc_imgsz, a.conf, a.target, True, a.grid) if a.acc_model else None
    eng = Engine(a, src, fast, acc)
    eng.event(f"Started: {src.desc}; watch={fast.label}; escalate={acc.label if acc else 'off'}")
    threading.Thread(target=eng.run, daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(eng))
    print(f"\n  NERISAL live is running.  Open  http://{local_ip()}:{a.port}  on the laptop.\n"
          f"  Logs: {a.logdir}/   Snapshots: {a.snapdir}/   Ctrl+C to stop.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        src.alive = False
        eng.logf.close()


if __name__ == "__main__":
    main()
