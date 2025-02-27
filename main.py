import cv2
import numpy as np
import time

# 4대의 카메라 스트림 URL (IP와 포트는 상황에 맞게 수정하세요)
camera_urls = [
    "http://10.0.0.81/stream",  # 정상 연결
    "http://10.0.0.82/stream",  # 연결 실패 시 재시도
    "http://10.0.0.83/stream",  # 연결 실패 시 재시도
    "http://10.0.0.84/stream"   # 연결 실패 시 재시도
]

# 각 카메라 정보를 저장 (URL과 VideoCapture 객체)
cameras = []
for url in camera_urls:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cameras.append({"url": url, "cap": cap, "last_try": time.time()})

# 각 분할 화면의 크기 (해상도에 맞게 조정)
frame_width, frame_height = 640, 480

def reinitialize_camera(camera):
    """카메라 재시도 함수: 캡처 객체를 재생성"""
    print(f"재접속 시도: {camera['url']}")
    # 간단히 캡처 객체를 해제한 후 다시 생성
    try:
        camera["cap"].release()
    except:
        pass
    camera["cap"] = cv2.VideoCapture(camera["url"], cv2.CAP_FFMPEG)
    camera["last_try"] = time.time()

def get_frame(camera):
    cap = camera["cap"]
    # 만약 캡처 객체가 열려있지 않다면 재시도 (최소 5초 간격)
    if not cap.isOpened() and (time.time() - camera["last_try"] > 5):
        reinitialize_camera(camera)
    try:
        ret, frame = cap.read()
    except Exception as e:
        print("프레임 읽기 예외:", e)
        ret = False
    if not ret or frame is None:
        # 프레임 읽기 실패 시 검은색 빈 프레임 반환
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (frame_width, frame_height))
    return frame

# 전체 화면 모드로 창 생성
cv2.namedWindow("4 Camera Streams", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("4 Camera Streams", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # 각 카메라에서 프레임 가져오기 (재시도 로직 포함)
    frames = [get_frame(cam) for cam in cameras]

    # 4분할 화면 구성: 상단 행과 하단 행
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    combined = cv2.vconcat([top_row, bottom_row])

    # 분할 화면 사이에 구분선 추가 (흰색 선, 두께 2)
    combined_with_lines = combined.copy()
    height, width = combined_with_lines.shape[:2]
    # 수직 구분선 (가로 가운데)
    cv2.line(combined_with_lines, (width//2, 0), (width//2, height), (255, 255, 255), thickness=2)
    # 수평 구분선 (세로 가운데)
    cv2.line(combined_with_lines, (0, height//2), (width, height//2), (255, 255, 255), thickness=2)

    cv2.imshow("4 Camera Streams", combined_with_lines)
    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 시 각 캡처 객체 해제
for cam in cameras:
    cam["cap"].release()
cv2.destroyAllWindows()
