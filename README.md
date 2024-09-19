# TensorFlow_ex

TensorFlow를 활용한 프로젝트를 진행하기 위하여 임시로 사용하는 레포지토리입니다.


- pip 버전 자체가 최신버전인지부터 우선 체크할 것

```
pip install tensorflow[and-cuda]
```

```
pip install --upgrade tensorflow keras
```


```
계획 예상

1. Tensor Flow의 Keras모델을 사용

2. ImageNet에서 약 분류별 500장 우선 학습 / 안되면 늘리기
    Class 종류
        1. 플라스틱컵 
        3. 휴지
        4. 비닐봉지
        6. 음료수병 >>>> 플라스틱
        7. 라면용기
        8. 캔음료

- 플라스틱, 일반, 캔, 비닐 으로 크게 4종류 분류

3. 테스트 정확도 95퍼 이상 확인


```