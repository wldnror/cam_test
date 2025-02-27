import cv2
import requests

# 예시: 1번 카메라 스트림 처리 및 사람 인식
cap = cv2.VideoCapture("http://10.0.0.82/stream")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # 사람 인식 알고리즘 (예: Haar Cascade, MobileNet-SSD 등)을 적용
    detected = person_detection(frame)  # 사용자 정의 함수
    if detected:
        # 사람이 감지되었으면 LED 켜기 요청
        requests.get("http://10.0.0.82/led_on")
        # 화면에 시각적 표시 (예: 사각형, 텍스트 등)
        cv2.putText(frame, "Person Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # 사람이 없으면 LED 끄기 요청
        requests.get("http://10.0.0.82/led_off")
    # 결과 출력
    cv2.imshow("Camera 1", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
