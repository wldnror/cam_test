import cv2
import numpy as np

# 4대의 카메라 스트림 URL (IP와 포트는 상황에 맞게 수정)
camera_urls = [
    "http://10.0.0.81/stream",  # 정상 연결됨
    "http://10.0.0.82/stream",  # 연결 실패
    "http://10.0.0.83/stream",  # 연결 실패
    "http://10.0.0.84/stream"   # 연결 실패
]

# VideoCapture 객체 생성 시 FFMPEG 백엔드를 사용 (가능하면)
caps = []
for url in camera_urls:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # 혹은 FFMPEG가 없으면 기본값 사용: cap = cv2.VideoCapture(url)
    caps.append(cap)

# 화면에 표시할 프레임 크기 (예: 640x480)
frame_width, frame_height = 640, 480

def get_frame(cap):
    # 캡처 객체가 열려있는지 먼저 확인
    if not cap.isOpened():
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    try:
        ret, frame = cap.read()
    except Exception as e:
        print("Exception during read:", e)
        ret = False
    if not ret or frame is None:
        # 프레임 읽기 실패 시 빈 프레임 반환
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    # 필요 시 프레임 크기 조절
    frame = cv2.resize(frame, (frame_width, frame_height))
    return frame

while True:
    # 각 캡처 객체에서 프레임 가져오기
    frames = [get_frame(cap) for cap in caps]

    # 4분할 화면 구성 (2행 2열)
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    combined = cv2.vconcat([top_row, bottom_row])

    cv2.imshow("4 Camera Streams", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 시 리소스 해제
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
