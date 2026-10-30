import mss
import cv2
import numpy as np
from ultralytics import YOLO
import time
import sys
import shutil
import subprocess
from pynput import keyboard
from pynput.mouse import Controller, Button


# ============================================================
#  НАСТРОЙКИ
# ============================================================

MODEL_PATH = '/home/user/Рабочий стол/cheremsha/runs/detect/cheremsha_clicker/weights/best.pt'

CLICK_BACKEND = 'pynput'

SETTLE = 0.008
HOLD   = 0.006

IMGSZ = None
REGION = None
MONITOR_INDEX = 1

NMS_IOU = 0.45
MAX_DET = 300
AGNOSTIC_NMS = False            # False = классы не подавляют друг друга

PRIORITY_ORDER = ['game', 'upgrade', 'ad']

# --- быстрое переключение game <-> upgrade ---
INTERVAL_CLASSES = {
    'upgrade': 0.8,             # было 4.0
    'ad': 2.0,
}

CONF_THRESHOLD = {
    'game': 0.30,
    'upgrade': 0.10,            # сильно понижен — ловим боковые ячейки
    'ad': 0.50,
}
DEFAULT_CONF = 0.40

CLICK_ALL = {'upgrade'}
MIN_DIST = 15                   # было 25

# --- интервал кликов game = 0.01 ---
CLICK_CONFIG = {
    'game': (8, 0.01),          # серия короче -> чаще возврат к upgrade
    'upgrade': (1, 0.05),
    'ad': (1, 0.40),
}

# --- запасной вариант: upgrade слева/справа от game ---
RELATIVE_UPGRADE = True
UP_OFFSET_X = 260               # расстояние от центра game по горизонтали
UP_OFFSET_Y = 0                 # смещение по вертикали, если нужно

DEBUG = True
SAVE_DEBUG = False              # True -> пишет debug.png с рамками


# ============================================================
#  ИНИЦИАЛИЗАЦИЯ
# ============================================================

model = YOLO(MODEL_PATH)

ID_TO_NAME = model.names
ID_TO_KEY = {i: str(n).strip().lower() for i, n in ID_TO_NAME.items()}
MODEL_KEYS = set(ID_TO_KEY.values())

ACTIVE_ORDER = [k for k in PRIORITY_ORDER if k in MODEL_KEYS]
ACTIVE_ORDER += [k for k in MODEL_KEYS if k not in ACTIVE_ORDER]

mouse = Controller()
HAS_XDOTOOL = shutil.which('xdotool') is not None

is_paused = False
is_running = True
last_click_time = {k: 0.0 for k in ACTIVE_ORDER}


# ============================================================
#  КЛИКИ
# ============================================================

def click_pynput(x, y):
    mouse.position = (x, y)
    time.sleep(SETTLE)
    mouse.press(Button.left)
    time.sleep(HOLD)
    mouse.release(Button.left)


def click_pynput_fast(x, y):
    mouse.press(Button.left)
    time.sleep(HOLD)
    mouse.release(Button.left)


def click_xdotool(x, y):
    subprocess.run(['xdotool', 'mousemove', str(x), str(y)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(SETTLE)
    subprocess.run(['xdotool', 'mousedown', '1'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(HOLD)
    subprocess.run(['xdotool', 'mouseup', '1'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def click_xdotool_raw(x, y):
    win = subprocess.run(['xdotool', 'getactivewindow'],
                         capture_output=True, text=True).stdout.strip()
    subprocess.run(['xdotool', 'mousemove', '--window', win, str(x), str(y),
                    'click', '--window', win, '1'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def click_at(x, y, move=True):
    if CLICK_BACKEND == 'xdotool' and HAS_XDOTOOL:
        click_xdotool(x, y)
    elif CLICK_BACKEND == 'xdotool_raw' and HAS_XDOTOOL:
        click_xdotool_raw(x, y)
    else:
        click_pynput(x, y) if move else click_pynput_fast(x, y)


# ============================================================
#  УПРАВЛЕНИЕ
# ============================================================

def on_press(key):
    global is_paused, is_running
    if key == keyboard.Key.space:
        is_paused = not is_paused
        if DEBUG:
            print('[PAUSE]' if is_paused else '[RESUME]')
    elif key == keyboard.Key.esc:
        is_running = False
        return False


listener = keyboard.Listener(on_press=on_press)
listener.daemon = True
listener.start()


# ============================================================
#  ЗАХВАТ
# ============================================================

def grab_screen():
    with mss.mss() as sct:
        mon = REGION if REGION else sct.monitors[MONITOR_INDEX]
        img = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
    return img, mon['left'], mon['top']


# ============================================================
#  ДЕТЕКЦИЯ
# ============================================================

def dedup(items, min_dist=MIN_DIST):
    kept = []
    for it in items:
        if all((it['x'] - k['x']) ** 2 + (it['y'] - k['y']) ** 2 > min_dist ** 2
               for k in kept):
            kept.append(it)
    return kept


def detect(img, ox, oy):
    kwargs = {'conf': 0.01, 'iou': NMS_IOU, 'max_det': MAX_DET,
              'agnostic_nms': AGNOSTIC_NMS, 'verbose': False}
    if IMGSZ:
        kwargs['imgsz'] = IMGSZ

    results = model(img, **kwargs)

    if SAVE_DEBUG:
        cv2.imwrite('debug.png', results[0].plot())

    grouped = {}
    raw_log = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            key = ID_TO_KEY.get(int(box.cls[0]), str(int(box.cls[0])))

            raw_log.append(f'{key}:{conf:.2f}')

            if conf < CONF_THRESHOLD.get(key, DEFAULT_CONF):
                continue

            grouped.setdefault(key, []).append({
                'key': key,
                'conf': conf,
                'x': int((x1 + x2) / 2) + ox,
                'y': int((y1 + y2) / 2) + oy,
                'virtual': False,
            })

    for k in list(grouped):
        grouped[k].sort(key=lambda d: d['conf'], reverse=True)
        grouped[k] = dedup(grouped[k])

    # --- достраиваем боковые upgrade относительно game ---
    if RELATIVE_UPGRADE and grouped.get('game'):
        g = grouped['game'][0]
        ups = grouped.get('upgrade', [])

        for dx in (-UP_OFFSET_X, UP_OFFSET_X):
            vx, vy = g['x'] + dx, g['y'] + UP_OFFSET_Y
            # если рядом уже есть реальная детекция — не дублируем
            if any((vx - u['x']) ** 2 + (vy - u['y']) ** 2 < (UP_OFFSET_X * 0.5) ** 2
                   for u in ups):
                continue
            ups.append({'key': 'upgrade', 'conf': 0.0,
                        'x': vx, 'y': vy, 'virtual': True})

        grouped['upgrade'] = ups

    # upgrade сортируем слева направо
    if grouped.get('upgrade'):
        grouped['upgrade'].sort(key=lambda d: d['x'])

    if DEBUG:
        counts = {k: len(v) for k, v in grouped.items() if v}
        print('raw:', ' '.join(raw_log) if raw_log else '—', '| targets:', counts)

    return grouped


# ============================================================
#  ОБРАБОТКА
# ============================================================

def handle_class(detections):
    if not detections:
        return False

    key = detections[0]['key']
    clicks, delay = CLICK_CONFIG.get(key, (1, 0.2))
    targets = detections if key in CLICK_ALL else detections[:1]

    for det in targets:
        if is_paused or not is_running:
            return True

        if DEBUG:
            mark = 'V' if det.get('virtual') else ' '
            print(f"  ->{mark}{key} ({det['x']}, {det['y']}) conf={det['conf']:.2f}")

        for i in range(clicks):
            if is_paused or not is_running:
                return True
            click_at(det['x'], det['y'], move=(i == 0))
            time.sleep(delay)

    last_click_time[key] = time.time()
    return True


def build_order(grouped):
    now = time.time()
    due, rest = [], []

    for key in ACTIVE_ORDER:
        if not grouped.get(key):
            continue
        interval = INTERVAL_CLASSES.get(key)
        if interval is not None and now - last_click_time.get(key, 0) >= interval:
            due.append(key)
        else:
            rest.append(key)

    return due + rest


def step():
    if is_paused:
        return False

    img, ox, oy = grab_screen()
    grouped = detect(img, ox, oy)

    for key in build_order(grouped):
        if is_paused or not is_running:
            return False
        if grouped.get(key):
            return handle_class(grouped[key])

    return False


# ============================================================
#  СТАРТ
# ============================================================

print('=' * 55)
print('Классы          :', ID_TO_NAME)
print('Порядок         :', ' -> '.join(ACTIVE_ORDER))
print('Интервалы       :', INTERVAL_CLASSES)
print('Относит. upgrade:', RELATIVE_UPGRADE, f'offset={UP_OFFSET_X}px')
print('ПРОБЕЛ — пауза  |  ESC — выход')
print('=' * 55)

try:
    while is_running:
        if is_paused:
            time.sleep(0.1)
        elif step():
            time.sleep(0.01)
        else:
            time.sleep(0.25)
except KeyboardInterrupt:
    pass
finally:
    listener.stop()
    sys.exit()