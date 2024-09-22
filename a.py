import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing import image

# 이미지 크기 및 배치 크기 설정
img_height = 244
img_width = 244
batch_size = 32

# 학습용 데이터셋 로드
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./image_data/",                                                      # 이미지 폴더 경로
  validation_split=0.2,                                                 # 20%는 검증용 데이터로 사용
  subset="training",                                                    # 학습용 데이터로 설정
  seed=123,                                                             # 랜덤 시드 고정
  image_size=(img_height, img_width),                                   # 이미지 크기 조정
)

# 검증용 데이터셋 로드
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  "./image_data/",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
)

class_names = train_ds.class_names
print(class_names)                                                      # 예시 출력: ['cats', 'dogs']


######################
######################


# 데이터 증강 레이어
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
])

# 학습 데이터셋 캐시 및 프리페치
# 검증 데이터셋 프리페치
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# 모델 정의
model = tf.keras.Sequential([
  data_augmentation,  # 데이터 증강
  layers.Conv2D(32, 3, activation='relu'),                                # 첫 번째 Convolution 층
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),                                # 두 번째 Convolution 층
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),                               # 세 번째 Convolution 층
  layers.MaxPooling2D(),
  layers.Flatten(),                                                       # 2D 데이터를 1D로 변환
  layers.Dense(128, activation='relu'),                                   # Fully Connected Layer
  layers.Dense(len(class_names), activation='softmax')                    # 출력층
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# 검증 정확도 확인
loss, accuracy = model.evaluate(val_ds)
print(f"Validation accuracy: {accuracy}")

# 새로운 이미지 예측
img = image.load_img('./Classification_Image/페트병.jpg', target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)                                  # 배치 차원 추가
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
print(f"Predicted class: {class_names[predicted_class]}")