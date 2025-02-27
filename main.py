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

# HOG 사람 검출기 초기화 (1번 카메라에 적용)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

window_name = "4 Camera Streams"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# 처음엔 전체화면 모드로 실행 (토글 가능)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

# 1번 카메라에서 사람 감지 결과를 저장할 변수
detection_active = False

while True:
    # 각 카메라의 최신 프레임을 가져옴 (스레드 안전하게)
    frames = []
    for cam in cameras:
        with cam["lock"]:
            frames.append(cam["frame"].copy())

    # 1번 카메라에 대해 사람 검출 (검출이 무겁다면 매 프레임이 아니라 주기적으로 처리할 수 있음)
    frame_cam1 = frames[0].copy()
    # 검출 결과: x, y, w, h 좌표 배열
    rects, weights = hog.detectMultiScale(frame_cam1, winStride=(8,8))
    # 감지 결과가 있다면 네모 표시
    if len(rects) > 0:
        detection_active = True
        for (x, y, w, h) in rects:
            cv2.rectangle(frame_cam1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        detection_active = False

    # 1번 카메라 프레임 업데이트 (검출된 결과 표시된 이미지로 교체)
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

    # 1번 카메라 영역에 대해 깜빡이는 테두리 (감지 시)
    # 1번 영역는 좌상단 영역 : x:0~quad_width, y:0~quad_height
    if detection_active:
        # 깜빡임 효과: 현재 시간 기반 주기로 색상 전환 (빨간색/투명)
        blink = int(time.time() * 2) % 2 == 0
        color = (0, 0, 255) if blink else (0, 0, 0)
        cv2.rectangle(combined_with_lines, (0, 0), (quad_width, quad_height), color, thickness=4)

    cv2.imshow(window_name, combined_with_lines)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키 누르면 종료
    if key == ord('q'):
        break
    # 'f' 키 누르면 전체화면/창 모드 토글
    if key == ord('f'):
        is_fullscreen = not is_fullscreen
        if is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cv2.destroyAllWindows()
