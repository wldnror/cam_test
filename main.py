import cv2
import numpy as np
import threading
import time

# 4대의 카메라 스트림 URL (1번은 실제 연결, 나머지는 연결이 안 된 상태)
camera_urls = [
    "http://10.0.0.81/stream",  # 1번: 실제 연결 (사람 검출 대상)
    "http://10.0.0.82/stream",  # 2번: 연결 없음 (재시도)
    "http://10.0.0.83/stream",  # 3번: 연결 없음 (재시도)
    "http://10.0.0.84/stream"   # 4번: 연결 없음 (재시도)
]

# 모니터 해상도 (16:9 예시: 1920×1080)
monitor_width, monitor_height = 1920, 1080
quad_width, quad_height = monitor_width // 2, monitor_height // 2  # 960x540

#########################
# 1번 카메라 (검출 대상)
#########################
cam1 = {
    "url": camera_urls[0],
    "cap": cv2.VideoCapture(camera_urls[0], cv2.CAP_FFMPEG),
    "frame": np.zeros((quad_height, quad_width, 3), dtype=np.uint8),
    "lock": threading.Lock(),
    "last_try": time.time(),
    "detection_active": False,
    "boxes": []  # 검출된 사람 사각형 리스트: (startX, startY, endX, endY)
}

#########################
# 2~4번 카메라 (단순 캡처)
#########################
other_cams = []
for url in camera_urls[1:]:
    cam = {
        "url": url,
        "cap": cv2.VideoCapture(url, cv2.CAP_FFMPEG),
        "frame": np.zeros((quad_height, quad_width, 3), dtype=np.uint8),
        "lock": threading.Lock(),
        "last_try": time.time()
    }
    other_cams.append(cam)

# 캡처 스레드 (카메라 연결 및 최신 프레임 업데이트)
def capture_thread(cam):
    while True:
        cap = cam["cap"]
        if not cap.isOpened():
            if time.time() - cam["last_try"] > 5:
                print(f"[재접속] {cam['url']}")
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
            # 분할 크기(960x540)로 리사이즈
            frame = cv2.resize(frame, (quad_width, quad_height))
            with cam["lock"]:
                cam["frame"] = frame
        else:
            with cam["lock"]:
                cam["frame"] = np.zeros((quad_height, quad_width, 3), dtype=np.uint8)
            time.sleep(0.1)
        time.sleep(0.01)

# 검출 스레드: 1번 카메라에서 최신 프레임을 이용해 MobileNet-SSD로 사람 검출
def detection_thread(cam):
    # 모델 파일들은 코드와 같은 디렉토리에 있어야 함
    prototxt = "MobileNetSSD_deploy.prototxt"
    model = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    # 최적화 (필요시 백엔드/타겟 설정)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # MobileNet-SSD 모델의 클래스 리스트
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    
    # 검출 입력 크기를 300x300으로 사용
    detection_width, detection_height = 300, 300
    # 스케일 팩터: 원본 분할 크기 / 검출 입력 크기
    scaleX = quad_width / detection_width
    scaleY = quad_height / detection_height

    while True:
        with cam["lock"]:
            frame = cam["frame"].copy()  # 원본: 960x540
        if frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        # 검출 입력으로 300x300 리사이즈
        resized = cv2.resize(frame, (detection_width, detection_height))
        blob = cv2.dnn.blobFromImage(resized, 0.007843, (detection_width, detection_height), 127.5)
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        detection_active = False

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    detection_active = True
                    box = detections[0, 0, i, 3:7]
                    # 검출 결과는 0~1의 비율값 → 300x300 기준 좌표로 변환 후, 원본 분할 크기로 확장
                    (startX, startY, endX, endY) = (box * np.array([detection_width, detection_height, detection_width, detection_height])).astype("int")
                    startX = int(startX * scaleX)
                    startY = int(startY * scaleY)
                    endX = int(endX * scaleX)
                    endY = int(endY * scaleY)
                    boxes.append((startX, startY, endX, endY))
        with cam["lock"]:
            cam["detection_active"] = detection_active
            cam["boxes"] = boxes
        time.sleep(0.02)  # 약 20ms 주기로 검출 (50fps 목표)

# 스레드 시작
t_cam1 = threading.Thread(target=capture_thread, args=(cam1,), daemon=True)
t_cam1.start()

t_det1 = threading.Thread(target=detection_thread, args=(cam1,), daemon=True)
t_det1.start()

threads = [t_cam1, t_det1]
for cam in other_cams:
    t = threading.Thread(target=capture_thread, args=(cam,), daemon=True)
    t.start()
    threads.append(t)

# 메인 루프: 전체 화면 구성 및 출력
window_name = "4 Camera Streams"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

while True:
    # 1번 카메라: 최신 프레임과 검출 결과
    with cam1["lock"]:
        frame1 = cam1["frame"].copy()
        detection_active = cam1["detection_active"]
        boxes = cam1["boxes"].copy()
    # 검출된 영역 사각형 표시
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(frame1, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    # 나머지 카메라 프레임 가져오기
    frames_other = []
    for cam in other_cams:
        with cam["lock"]:
            frames_other.append(cam["frame"].copy())
    
    # 2행 2열 분할: 1번은 좌상단, 나머지는 순서대로 배치
    top_row = cv2.hconcat([frame1, frames_other[0]])
    bottom_row = cv2.hconcat([frames_other[1], frames_other[2]])
    combined = cv2.vconcat([top_row, bottom_row])
    
    # 구분선 추가 (흰색, 두께 2)
    combined_with_lines = combined.copy()
    h, w = combined_with_lines.shape[:2]
    cv2.line(combined_with_lines, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=2)
    cv2.line(combined_with_lines, (0, h // 2), (w, h // 2), (255, 255, 255), thickness=2)
    
    # 1번 카메라 영역 (좌상단)에 대해 사람이 감지되면 깜빡이는 테두리 표시
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
