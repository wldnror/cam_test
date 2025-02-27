import cv2
import numpy as np
import time

# 4대의 카메라 스트림 URL (IP와 포트는 상황에 맞게 수정)
camera_urls = [
    "http://10.0.0.81/stream",  # 정상 연결
    "http://10.0.0.82/stream",  # 연결 실패 시 재시도
    "http://10.0.0.83/stream",  # 연결 실패 시 재시도
    "http://10.0.0.84/stream"   # 연결 실패 시 재시도
]

# 각 카메라 정보를 저장 (URL, VideoCapture 객체, 마지막 재시도 시간)
cameras = []
for url in camera_urls:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cameras.append({"url": url, "cap": cap, "last_try": time.time()})

# 모니터 해상도 (16:9 예시: 1920x1080)
monitor_width, monitor_height = 1920, 1080
# 2x2 그리드이므로 각 분할 화면 크기
quad_width, quad_height = monitor_width // 2, monitor_height // 2

# 전체화면 여부 토글 변수
is_fullscreen = True
window_name = "4 Camera Streams"

def reinitialize_camera(camera):
    """캠 재접속 함수: 캡처 객체를 재생성"""
    print(f"재접속 시도: {camera['url']}")
    try:
        camera["cap"].release()
    except Exception as e:
        print("캡 해제 에러:", e)
    camera["cap"] = cv2.VideoCapture(camera["url"], cv2.CAP_FFMPEG)
    camera["last_try"] = time.time()

def get_frame(camera):
    cap = camera["cap"]
    # 캡처가 열려있지 않거나 마지막 재시도 후 5초가 지난 경우 재시도
    if not cap.isOpened() and (time.time() - camera["last_try"] > 5):
        reinitialize_camera(camera)
    try:
        ret, frame = cap.read()
    except Exception as e:
        print("프레임 읽기 예외:", e)
        ret = False
    if not ret or frame is None:
        # 읽기 실패 시 검은색 빈 프레임 반환
        frame = np.zeros((quad_height, quad_width, 3), dtype=np.uint8)
    else:
        # 프레임 크기를 16:9 분할에 맞게 조정 (960x540)
        frame = cv2.resize(frame, (quad_width, quad_height))
    return frame

# 창 생성 (처음엔 전체화면)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # 각 카메라에서 프레임 얻기 (재시도 포함)
    frames = [get_frame(cam) for cam in cameras]

    # 2행 2열 분할 구성
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    combined = cv2.vconcat([top_row, bottom_row])

    # 구분선 추가 (흰색 선, 두께 2)
    combined_with_lines = combined.copy()
    h, w = combined_with_lines.shape[:2]
    cv2.line(combined_with_lines, (w // 2, 0), (w // 2, h), (255, 255, 255), thickness=2)
    cv2.line(combined_with_lines, (0, h // 2), (w, h // 2), (255, 255, 255), thickness=2)

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

# 종료 시 각 캡처 객체 해제
for cam in cameras:
    cam["cap"].release()
cv2.destroyAllWindows()
