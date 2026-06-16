import cv2  # OpenCV 영상 처리 라이브러리 임포트
import os  # 운영체제 관련 기능(경로 설정 등) 임포트
import requests  # HTTP 통신(텔레그램 API 전송) 라이브러리 임포트
import time  # 시간 지연 및 타임스탬프 생성을 위한 모듈 임포트
import threading  # 비동기 처리를 위한 멀티스레딩 모듈 임포트
from gpiozero import LED  # 라즈베리 파이 GPIO 제어 라이브러리 임포트

TELEGRAM_TOKEN = '8725076847:AAH-IyftP1Tt-v6xAe3x0Y9UHQrZ3Thcs0k'  # 텔레그램 봇 토큰 설정
CHAT_ID = '8197602658'  # 메시지를 받을 텔레그램 채팅 ID 설정

green_led = LED(2)  # GPIO 2번 핀에 초록색 LED 할당 (남성용)
red_led = LED(3)  # GPIO 3번 핀에 빨간색 LED 할당 (여성용)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 실행 파일의 절대 경로 저장

face_cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_alt.xml')  # 얼굴 인식 모델 경로 설정
cascade = cv2.CascadeClassifier(face_cascade_path)  # 얼굴 인식 분류기 객체 생성

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # 딥러닝 모델 학습 시 사용된 평균 RGB 값

age_proto_path = os.path.join(BASE_DIR, 'deploy_age.prototxt')  # 나이 추정 네트워크 구조 파일 경로
age_model_path = os.path.join(BASE_DIR, 'age_net.caffemodel')  # 나이 추정 모델 가중치 파일 경로

gender_proto_path = os.path.join(BASE_DIR, 'deploy_gender.prototxt')  # 성별 추정 네트워크 구조 파일 경로
gender_model_path = os.path.join(BASE_DIR, 'gender_net.caffemodel')  # 성별 추정 모델 가중치 파일 경로

age_net = cv2.dnn.readNetFromCaffe(age_proto_path, age_model_path)  # Caffe 나이 모델 로드
gender_net = cv2.dnn.readNetFromCaffe(gender_proto_path, gender_model_path)  # Caffe 성별 모델 로드

age_list = ['(0 ~ 2)','(4 ~ 6)','(8 ~ 12)','(15 ~ 20)','(25 ~ 32)','(38 ~ 43)','(48 ~ 53)','(60 ~ 100)']  # 나이 분류 리스트
gender_list = ['Male', 'Female']  # 성별 분류 리스트

def process_alert(photo_path, gender_label, age_label):  # 텔레그램 전송 및 하드웨어 제어용 함수
    if gender_label == 'Male':  # 남성일 경우
        green_led.on()  # 초록 LED 점등
        print("[알림] 남자 감지: 초록색 LED 점등")
    else:  # 여성일 경우
        red_led.on()  # 빨간 LED 점등
        print("[알림] 여자 감지: 빨간색 LED 점등")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"  # 텔레그램 사진 전송 API 주소
    info_text = f"새로운 인물 감지!\n성별: {gender_label}\n추정 나이: {age_label}"  # 전송할 메시지 내용
    
    try:  # 예외 처리 시작
        with open(photo_path, 'rb') as photo:  # 촬영한 사진 파일을 읽기 모드로 열기
            payload = {'chat_id': CHAT_ID, 'caption': info_text}  # 채팅 ID와 메시지 설정
            files = {'photo': photo}  # 전송할 파일 설정
            response = requests.post(url, data=payload, files=files)  # API로 사진 전송 요청
            print(">>> 텔레그램 응답 결과:", response.json())  # 응답 결과 출력
    except Exception as e:  # 전송 오류 발생 시
        print(f"텔레그램 전송 중 예외 발생: {e}")  # 에러 메시지 출력
    finally:  # 작업 완료 후 항상 실행
        if os.path.exists(photo_path):  # 사진 파일이 존재하면
            os.remove(photo_path)  # 로컬 사진 파일 삭제

    time.sleep(3)  # LED 3초간 켜짐 유지
    green_led.off()  # LED 끄기
    red_led.off()  # LED 끄기

def main():  # 메인 실행 함수
    camera = cv2.VideoCapture(-1)  # 카메라 장치 열기
    camera.set(3, 640)  # 가로 해상도 640 설정
    camera.set(4, 480)  # 세로 해상도 480 설정

    last_capture_time = 0  # 마지막 캡처 시간 초기화
    cooldown_seconds = 10  # 알림 쿨타임(10초) 설정

    print("나이/성별 인식 보안 시스템을 시작합니다...")  # 시작 메시지 출력

    while(camera.isOpened()):  # 카메라가 열려있는 동안 반복
        ret, img = camera.read()  # 카메라로부터 한 프레임 읽기
        if not ret: continue  # 읽기 실패 시 건너뜀

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백으로 변환(얼굴 인식 정확도 향상)
        results = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))  # 얼굴 영역 탐지

        for box in results:  # 탐지된 얼굴마다 반복
            x, y, w, h = box  # 좌표 및 크기 할당
            face = img[int(y):int(y+h), int(x):int(x+w)].copy()  # 얼굴 영역만 잘라내기

            if face.shape[0] == 0 or face.shape[1] == 0: continue  # 영역이 비었으면 건너뜀

            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)  # 딥러닝 입력을 위한 Blob 변환

            gender_net.setInput(blob)  # 성별 네트워크 입력
            gender_preds = gender_net.forward()  # 성별 추론 결과값 산출
            gender_idx = gender_preds.argmax()  # 가장 높은 확률의 성별 인덱스 추출
            gender_label = gender_list[gender_idx]  # 성별 레이블 가져오기

            age_net.setInput(blob)  # 나이 네트워크 입력
            age_preds = age_net.forward()  # 나이 추론 결과값 산출
            age_idx = age_preds.argmax()  # 가장 높은 확률의 나이대 인덱스 추출
            age_label = age_list[age_idx]  # 나이 레이블 가져오기
            
            info = f"{gender_label} {age_label}"  # 결과 정보 문자열 생성

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), thickness=2)  # 얼굴 영역 사각형 그리기
            cv2.putText(img, info, (x, y-15), 0, 0.5, (0, 255, 0), 1)  # 결과 정보 텍스트 표시

            current_time = time.time()  # 현재 시간 측정
            if current_time - last_capture_time > cooldown_seconds:  # 쿨타임 경과 여부 확인
                filename = f"detected_{int(current_time)}.jpg"  # 파일명 생성
                cv2.imwrite(filename, img)  # 현재 화면 파일로 저장

                alert_thread = threading.Thread(target=process_alert, args=(filename, gender_label, age_label))  # 텔레그램 전송 스레드 생성
                alert_thread.daemon = True  # 데몬 스레드 설정(프로그램 종료 시 같이 종료)
                alert_thread.start()  # 알림 스레드 시작
                
                last_capture_time = current_time  # 마지막 캡처 시간 업데이트

        cv2.imshow('Security Camera', img)  # 결과 화면 보여주기
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키 누르면 종료
            break

    camera.release()  # 카메라 리소스 해제
    cv2.destroyAllWindows()  # 모든 창 닫기
    green_led.off()  # LED 끄기
    red_led.off()  # LED 끄기

if __name__ == '__main__':  # 실행 파일일 경우
    main()  # 메인 함수 호출