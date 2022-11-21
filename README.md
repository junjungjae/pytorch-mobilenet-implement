# pytorch-mobilenet-implementation

### Refernce
1. https://deep-learning-study.tistory.com/549
2. MobileNet 논문(HOWARD, Andrew G., et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861, 2017.)


### 구현 & 학습결과
- 확실히 이전에 구현했던 모델보다는 모델 사이즈의 경량화가 잘 되어있음.
- 과적합이 너무 심함. 아래 항목은 과적합 이유 추측
    1. 이미지 전처리 과정의 부재(주요 원인 중 하나)
    2. 생각보다 심한 정보의 소실(하지만 이정도로 심하면 타 모델에서 depthwise 기능을 썼을리 없음. 보류)
    3. Dropout 등 과적합 방지 기능의 부재(주요 원인 중 하나)
    4. 데이터셋 고유 문제 or train / val dataset 간의 불균형
        - 추가 실험이 필요하겠지만 원본 or train/val 데이터셋의 문제는 아닐듯함.

### 추후 개선점
- 이미지 전처리 기능 추가
- 로그 추가
- 기능 세분화(데이터 로딩, 전처리, 초기화, 학습...)
