##### 실행 #####
# 이미지를 저장하지 않을 경우
# sudo python3 object_detection_image.py --input 이미지 경로
# 예) sudo python3 object_detection_image.py --input ./test_image/test_image_1.jpg
#
# 이미지를 저장할 경우
# sudo python3 object_detection_image.py --input 이미지 경로 --output 저장할 이미지 경로
# 예) sudo python3 object_detection_image.py --input ./test_image/test_image_1.jpg --output ./result_image/result_image_1.jpg

# 필요한 패키지 import
import numpy as np # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import argparse # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import cv2 # opencv 모듈

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser() # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", required=True, help="input 이미지 경로")
ap.add_argument("-o", "--output", type=str, help="output 이미지 경로") # 이미지 저장 경로
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="최소 확률")
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# 훈련된 클래스 labels 목록을 초기화
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 각 클래스에 대한 bounding box 색깔 random 지정
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 모델(caffemodel 및 prototxt) load
print("[모델 loading...]")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# input 이미지 읽기
image = cv2.imread(args["input"])

# 이미지 크기
(h, w) = image.shape[:2]

# blob 이미지 생성
# 파라미터
# 1) image : 사용할 이미지
# 2) scalefactor : 이미지 크기 비율 지정
# 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
# 4) mean : Mean Subtraction 값을 RGB 색상 채널별로 지정해 주는 경험치 값(최적의 값)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# 객체 인식
print("[객체 인식]")
net.setInput(blob)
detections = net.forward() # Caffe 모델이 처리한 결과값(4차원 배열)

# 객체 인식 수 만큼 반복
for i in np.arange(0, detections.shape[2]):
    # 객체 확률 추출
    confidence = detections[0, 0, i, 2]
    
    # 객체 확률이 최소 확률보다 큰 경우
    if confidence > args["confidence"]:
        # 인식된 객체 index
        idx = int(detections[0, 0, i, 1])

        # bounding box 위치 계산
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # bounding box 출력
        cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
        
        # 객체 인식된 클래스 label 및 확률
        label = "{} : {:.2f}%".format(CLASSES[idx], confidence * 100)
        print("[{}]".format(label))
        
        # label text 잘림 방지
        y = startY - 15 if startY - 15 > 15 else startY + 15
        
        # label text 출력
        cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 3)

# 이미지 저장
if args["output"] !=  None: # output 이미지 경로를 입력하였을 때(입력하지 않은 경우 저장되지 않음)
    cv2.imwrite(args["output"], image) # 파일로 저장, 포맷은 확장자에 따름

# 이미지 show
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
