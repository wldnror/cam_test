import cv2
import numpy as np
import threading
import time

# 4대의 카메라 스트림 URL (각 카메라가 동일 조건으로 동작)
camera_urls = [
    "http://10.0.0.81/stream",  # 1번 카메라
    "http://10.0.0.82/stream",  # 2번 카메라
    "http://10.0.0.83/stream",  # 3번 카메라
    "http://10.0.0.84/stream"   # 4번 카메라
]

# 모니터 해상도 (16:9 예시: 1920×1080)
monitor_width, monitor_height = 1920, 1080
quad_width, quad_height = monitor_width // 2, monitor_height // 2  # 각 분할: 960×540

# 각 카메라 객체 생성 (캡처, 최신 프레임, 검출 결과 등 포함)
cameras = []
for url in camera_urls:
    cam = {
        "url": url,
        "cap": cv2.VideoCapture(url, cv2.CAP_FFMPEG),
        "frame": np.zeros((quad_height, quad_width, 3), dtype=np.uint8),
        "lock": threading.Lock(),
        "last_try": time.time(),
        "detection_active": False,
        "boxes": []  # 검출된 사람 사각형 리스트: [(startX, startY, endX, endY), ...]
    }
    cameras.append(cam)

#########################
# 캡처 스레드 (모든 카메라 공통)
#########################
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

#########################
# 검출 스레드 (모든 카메라에 대해 동일하게 적용)
#########################
def detection_thread(cam):
    # 모델 파일들은 코드와 같은 디렉토리에 있어야 함
    prototxt = "MobileNetSSD_deploy.prototxt"
    model = "MobileNetSSD_deploy.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # MobileNet-SSD 클래스 리스트
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    
    # 검출 입력 크기를 300x300으로 사용
    detection_width, detection_height = 10, 10
    # 스케일 팩터: 원본 분할 크기 / 검출 입력 크기
    scaleX = quad_width / detection_width
    scaleY = quad_height / detection_height

    while True:
        with cam["lock"]:
            frame = cam["frame"].copy()  # 원본: 960x540
        if frame is None or frame.size == 0:
            time.sleep(0.01)
            continue

        # 검출용 입력: 300x300으로 리사이즈
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
                    # 결과는 0~1의 비율값이므로, 300x300 기준 좌표로 변환 후 원본 크기로 확장
                    (startX, startY, endX, endY) = (box * np.array([detection_width, detection_height, detection_width, detection_height])).astype("int")
                    startX = int(startX * scaleX)
                    startY = int(startY * scaleY)
                    endX = int(endX * scaleX)
                    endY = int(endY * scaleY)
                    boxes.append((startX, startY, endX, endY))
        with cam["lock"]:
            cam["detection_active"] = detection_active
            cam["boxes"] = boxes
        time.sleep(0.02)  # 약 20ms 주기 (필요시 조정)

#########################
# 스레드 시작 (각 카메라마다 캡처+검출)
#########################
threads = []
for cam in cameras:
    t_cap = threading.Thread(target=capture_thread, args=(cam,), daemon=True)
    t_cap.start()
    threads.append(t_cap)
    
    t_det = threading.Thread(target=detection_thread, args=(cam,), daemon=True)
    t_det.start()
    threads.append(t_det)

#########################
# 메인 루프: 2x2 분할 화면 구성 및 출력
#########################
window_name = "4 Camera Streams"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

while True:
    frames = []
    # 각 카메라의 최신 프레임과 검출 결과 반영
    for cam in cameras:
        with cam["lock"]:
            frame = cam["frame"].copy()
            # 검출된 영역 표시
            for (startX, startY, endX, endY) in cam["boxes"]:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            frames.append(frame)
    
    # 2행 2열 분할 구성
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    combined = cv2.vconcat([top_row, bottom_row])
    
    # 구분선 추가 (흰색, 두께 2)
    combined_with_lines = combined.copy()
    h, w = combined_with_lines.shape[:2]
    cv2.line(combined_with_lines, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=2)
    cv2.line(combined_with_lines, (0, h // 2), (w, h // 2), (255, 255, 255), thickness=2)
    
    # 각 카메라 영역에 대해 사람이 감지되면 해당 영역에 깜빡이는 테두리 표시
    # 각 분할 영역 좌표: 
    # 1번: (0,0)-(quad_width,quad_height)
    # 2번: (quad_width,0)-(w,quad_height)
    # 3번: (0,quad_height)-(quad_width, h)
    # 4번: (quad_width,quad_height)-(w, h)
    for i, (x, y) in enumerate([(0,0), (quad_width,0), (0,quad_height), (quad_width,quad_height)]):
        if cameras[i]["detection_active"]:
            blink = int(time.time() * 2) % 2 == 0
            color = (0, 0, 255) if blink else (0, 0, 0)
            cv2.rectangle(combined_with_lines, (x, y), (x+quad_width, y+quad_height), color, thickness=4)
    
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
