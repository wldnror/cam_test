import cv2
import numpy as np
import threading
import time

# 카메라 URL (1번은 정상 연결, 나머지는 테스트용)
camera_urls = [
    "http://10.0.0.81/stream",
    "http://10.0.0.82/stream",
    "http://10.0.0.83/stream",
    "http://10.0.0.84/stream"
]

# 모니터 해상도 (16:9 예시: 1920x1080)
monitor_width, monitor_height = 1920, 1080
quad_width, quad_height = monitor_width // 2, monitor_height // 2

# 각 카메라 정보를 저장
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
    while True:
        cap = cam["cap"]
        # 캡처 객체 재접속 로직
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

        # 버퍼 플러싱: 최신 프레임만 읽도록 여러 번 grab() 호출
        flush_frames = 5  # 상황에 따라 조정 가능
        for _ in range(flush_frames):
            cap.grab()
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

# 각 카메라 별 스레드 생성 및 시작
threads = []
for cam in cameras:
    t = threading.Thread(target=capture_thread, args=(cam,), daemon=True)
    t.start()
    threads.append(t)

# (이전 코드와 이어서 사람 검출 등 처리)
# 예시: MobileNet-SSD 기반 검출 코드는 이전 예제를 참고하세요.
# 여기서는 버퍼링 문제 개선을 위한 캡처 스레드 수정에 집중합니다.

window_name = "4 Camera Streams"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
is_fullscreen = True

while True:
    frames = []
    for cam in cameras:
        with cam["lock"]:
            frames.append(cam["frame"].copy())
    
    # 예제에서는 단순히 4분할 화면을 구성합니다.
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    combined = cv2.vconcat([top_row, bottom_row])
    
    # 구분선 추가
    combined_with_lines = combined.copy()
    h, w = combined_with_lines.shape[:2]
    cv2.line(combined_with_lines, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=2)
    cv2.line(combined_with_lines, (0, h // 2), (w, h // 2), (255, 255, 255), thickness=2)
    
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
