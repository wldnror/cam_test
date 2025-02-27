import cv2
import numpy as np
import threading
import time

# 4대의 카메라 스트림 URL (실제 연결되는 것은 1번만 있고, 나머지는 없는 상태)
camera_urls = [
    "http://10.0.0.81/stream",  # 정상 연결된 카메라 (사람 감지 대상)
    "http://10.0.0.82/stream",  # 연결 없음 (재시도)
    "http://10.0.0.83/stream",  # 연결 없음 (재시도)
    "http://10.0.0.84/stream"   # 연결 없음 (재시도)
]

# 모니터 해상도 (16:9 예시: 1920x1080)
monitor_width, monitor_height = 1920, 1080
quad_width, quad_height = monitor_width // 2, monitor_height // 2

# 각 카메라의 최신 프레임과 캡처 객체, 스레드를 저장할 리스트
cameras = []
for url in camera_urls:
    cam = {
        "url": url,
        "cap": cv2.VideoCapture(url, cv2.CAP_FFMPEG),
        "frame": np.zeros((quad_height, quad_width, 3), dtype=np.uint8),
        "lock": threading.Lock(),
        "last_try": time.time()
    }
    cameras.append(cam)

def capture_thread(cam):
    """각 카메라의 프레임을 계속 읽어오는 스레드 함수"""
    while True:
        cap = cam["cap"]
        # 캡처 객체가 열려있지 않으면 일정 간격 후 재접속 시도
        if not cap.isOpened():
            if time.time() - cam["last_try"] > 5:
                print(f"재접속 시도: {cam['url']}")
                try:
                    cap.release()
                except Exception as e:
                    print("캡 해제 에러:", e)
                cam["cap"] = cv2.VideoCapture(cam["url"], cv2.CAP_FFMPEG)
                cam["last_try"] = time.time()
            time.sleep(0.5)
            continue

        try:
            ret, frame = cap.read()
        except Exception as e:
            print("프레임 읽기 예외:", e)
            ret = False

        if ret and frame is not None:
            frame = cv2.resize(frame, (quad_width, quad_height))
            with cam["lock"]:
                cam["frame"] = frame
        else:
            with cam["lock"]:
                cam["frame"] = np.zeros((quad_height, quad_width, 3), dtype=np.uint8)
            time.sleep(0.1)
        time.sleep(0.01)

# 각 카메라마다 스레드 생성 및 시작
threads = []
for cam in cameras:
    t = threading.Thread(target=capture_thread, args=(cam,), daemon=True)
    t.start()
    threads.append(t)

# MobileNet-SSD 모델 로드 (1번 카메라 검출용)
prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# 검출할 클래스: MobileNet-SSD 모델의 클래스 리스트
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

window_name = "4 Camera Streams"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

detection_active = False

while True:
    # 각 카메라의 최신 프레임을 가져옴 (스레드 안전하게)
    frames = []
    for cam in cameras:
        with cam["lock"]:
            frames.append(cam["frame"].copy())

    # MobileNet-SSD를 이용해 1번 카메라에서 사람 감지
    frame_cam1 = frames[0].copy()
    blob = cv2.dnn.blobFromImage(frame_cam1, 0.007843, (quad_width, quad_height), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detection_active = False
    # 탐지 결과 반복
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 신뢰도 임계값
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                detection_active = True
                box = detections[0, 0, i, 3:7] * np.array([quad_width, quad_height, quad_width, quad_height])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame_cam1, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 1번 카메라 프레임 업데이트 (검출 결과 표시)
    frames[0] = frame_cam1

    # 2행 2열 분할 구성
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    combined = cv2.vconcat([top_row, bottom_row])

    # 구분선 추가 (흰색 선, 두께 2)
    combined_with_lines = combined.copy()
    h, w = combined_with_lines.shape[:2]
    cv2.line(combined_with_lines, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=2)
    cv2.line(combined_with_lines, (0, h // 2), (w, h // 2), (255, 255, 255), thickness=2)

    # 1번 카메라 영역 (좌상단)에 대해 깜빡이는 테두리 표시 (감지 시)
    if detection_active:
        blink = int(time.time() * 2) % 2 == 0
        color = (0, 0, 255) if blink else (0, 0, 0)
        cv2.rectangle(combined_with_lines, (0, 0), (quad_width, quad_height), color, thickness=4)

    cv2.imshow(window_name, combined_with_lines)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cv2.destroyAllWindows()
