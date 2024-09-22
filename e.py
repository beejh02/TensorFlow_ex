import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# VGG16 모델 로드 (ImageNet 가중치 사용, 최종 분류 레이어 제외)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# VGG16의 기존 가중치를 고정하여 학습되지 않도록 설정
for layer in base_model.layers:
    layer.trainable = False

# Sequential 모델 정의
model = models.Sequential()

# base_model을 Sequential 모델에 추가
model.add(base_model)

# Flatten과 추가적인 Dense 레이어 추가
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))  # 이진 분류의 경우

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 전처리 (ImageDataGenerator를 이용한 데이터 증강)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './image_data/',  # 학습 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # 이진 분류 예시 (다중 분류는 'categorical' 사용)
)

validation_generator = test_datagen.flow_from_directory(
    './Classification_Image/',  # 검증 데이터 경로
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# 학습 결과 평가
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
