import cv2
import numpy as np

# 4대의 카메라 스트림 URL (IP와 포트는 상황에 맞게 수정하세요)
camera_urls = [
    "http://10.0.0.81/stream",
    "http://10.0.0.82/stream",
    "http://10.0.0.83/stream",
    "http://10.0.0.84/stream"
]

# 각 카메라 VideoCapture 객체 생성
caps = [cv2.VideoCapture(url) for url in camera_urls]

# 모니터 화면의 해상도에 맞춰 빈 프레임의 크기 지정 (예: 640x480)
frame_width, frame_height = 640, 480

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        # 프레임 읽기에 실패하면 검은색 빈 프레임 반환
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    else:
        # 필요시 프레임 크기 조절
        frame = cv2.resize(frame, (frame_width, frame_height))
    return frame

while True:
    frames = [get_frame(cap) for cap in caps]

    # 4분할 화면 구성: 두 행, 두 열
    top_row = cv2.hconcat([frames[0], frames[1]])
    bottom_row = cv2.hconcat([frames[2], frames[3]])
    combined = cv2.vconcat([top_row, bottom_row])

    cv2.imshow("4 Camera Streams", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
