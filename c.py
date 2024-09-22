import tensorflow as tf
import numpy as np
import kagglehub
from tensorflow.keras.preprocessing import image

# 모델 다운로드
path = kagglehub.model_download("tensorflow/efficientnet/tfLite/lite0-fp32")
model_path = f"{path}/2.tflite"  # TFLite 모델 경로 설정
print("Path to model files:", model_path)

# 학습용 데이터셋 로드
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./image_data/",                                                      # 이미지 폴더 경로
)

class_names = train_ds.class_names
print(class_names)    

# TensorFlow Lite 인터프리터 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# 모델의 입력/출력 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 이미지 크기 설정 (EfficientNet Lite0은 224x224 크기를 사용)
IMAGE_SHAPE = (224, 224)

# 이미지를 로드하고 전처리하는 함수 정의
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SHAPE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = img_array / 255.0  # 이미지를 [0, 1] 범위로 정규화
    return img_array

# 이미지를 전처리하여 모델에 입력
img_path = './검증이미지/찌그러진캔.jpg'  # 예측할 이미지 경로
input_data = preprocess_image(img_path)

# 입력 데이터를 인터프리터에 전달
interpreter.set_tensor(input_details[0]['index'], input_data)

# 모델 추론 실행
interpreter.invoke()

# 예측 결과 가져오기
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

# 예측된 클래스 인덱스 출력
print(f"Predicted class index: {predicted_class}")


# 예측된 클래스 이름 출력
predicted_class_name = class_names[predicted_class]
print(f"Predicted class name: {predicted_class_name}")
