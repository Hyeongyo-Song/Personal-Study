**📌 이 자료는 필자와 같이 인공지능에 입문하고자 하는 분들에게 조금이라도 도움이 되길 바라며 작성되었습니다. 📌**
시간을 들여 **내용과 양식**을 꾸준히 개선해 나가겠습니다.  

**마지막 수정:** 2024/11/20
**작성자:** 송현교  

---

# 📚 AI 및 머신러닝 개념 정리

## 1️⃣ 인공지능이란?
기존의 규칙기반 프로그램을 넘어, 인간의 신경망을 본따 만든 인공신경망을 통해 스스로 의사결정하고 예측할 수 있는 프로그램

## 2️⃣ 머신러닝이란?
모델이 데이터로부터 유의미한 패턴을 추출하고 학습하는 방법을 총체적으로 일컫는 말.

## 3️⃣ 딥러닝이란?
인간의 신경망 구조를 본따 만든 인공신경망 개념을 기반으로 데이터의 유의미한 패턴을 추출, 분석, 학습하는 방법.

## 4️⃣ Supervised Learning
지도 학습은 모델이 데이터 뿐만 아니라 데이터의 정답 정보(Label)까지 함께 학습에 사용하는 학습이다. Classification, Regression 모델들이 일반적으로 지도 학습에 속함.

## 5️⃣ Classification
분류는 새로운 데이터가 들어왔을 때 해당 데이터가 어느 클래스에 속하는지 모델이 예측하는 것을 의미.  
- 두 개의 클래스 사이에서 예측 시: **Binary Classification**  
- 다중 클래스 분류 시: **Categorical Classification**  
대표적인 분류 모델: Logistic Regression, Support Vector Machine.

## 6️⃣ Logistic Regression
로지스틱 회귀는 이름과 달리 Binary Classification을 수행하는 모델이다.  
- **Logit**: Log-Odds (q/1-q 계산)  
- Sigmoid 활성화 함수를 통해 0~1 사이로 정규화된 값을 출력  
- 손실함수: **Binary Cross Entropy Loss**  
- 최적화 방향으로 학습 진행.

## 7️⃣ Support Vector Machine
서포트 벡터 머신은 Binary Classification을 수행하는 모델이다.  
- 두 클래스를 가장 잘 분리하는 결정 경계와 기울기를 찾는 것이 목표.  
- **라그랑주 승수법**: 제약 조건 기반 최적화 방법.  
- Support Vector 간의 마진을 최대화하여 결정 경계 학습.  
- **Kernel Trick**: 비선형 데이터 분리를 위한 고차원 변환.

## 8️⃣ Activation Function
입력값을 특정 형태로 변환하여 출력값으로 반환하는 함수.  
- **Sigmoid Function**: 0~1 사이 확률값 출력  
- **Softmax Function**: 벡터를 확률값으로 변환  
- **Tanh Function**: -1~1 정규화  
- **ReLU Function**: 음수는 0, 양수는 그대로 출력.

## 9️⃣ Dying ReLU
모든 입력값이 음수일 경우 출력값이 0이 되어 학습이 멈추는 현상.  
- 해결책: Exponential ReLU, Leaky ReLU, GELU 등 개선된 함수 사용.

## 🔟 Exponential Rectified Linear Unit
음수 값에 Exponential 연산으로 부드러운 음수 처리.  
- 양수 처리는 ReLU와 동일.

## 1️⃣1️⃣ Leaky Rectified Linear Unit
작은 음수(예: 0.01 등)는 허용하여 학습이 멈추는 것을 방지.  
- 가장 일반적으로 사용되는 활성화 함수.

## 1️⃣2️⃣ Cross Entropy Loss
실제 Label과 모델 예측값의 확률분포 차이를 계산하는 손실 함수.  
- **Binary Classification**: Binary Cross Entropy Loss  
- **Multi-Class Classification**: Categorical Cross Entropy Loss.

## 1️⃣3️⃣ Regression
독립변수 x로 종속변수 y를 예측.  
- **Linear Regression**: 선형 결과  
- **Simple/Multi Linear Regression**: 직선/평면  
- **Non-Linear Regression**: 비선형 결과.

## 1️⃣4️⃣ K-Neariest Neighborhood
지도 학습 기반 거리 기반 Classification 모델.  
- 새로운 데이터는 가장 가까운 K개의 이웃 데이터를 기반으로 그룹 배치.

## 1️⃣5️⃣ Unsupervised Learning
정답 정보(Label) 없이 데이터만으로 학습하는 방법.  
- 데이터에서 유의미한 패턴을 스스로 추출하여 정의.  
- **Clustering**: 대표적인 비지도 학습 기법.

## 1️⃣6️⃣ Clustering
유사한 패턴의 데이터들을 그룹으로 묶는 기법.  
- 데이터를 그룹화하여 학습.

## 1️⃣7️⃣ K-Means Clustering
유사한 패턴의 데이터를 K개의 그룹으로 군집화.  
- **Centroid**: 각 그룹의 중심  
- **Silhouette Coefficient**: 0~1로 군집화 정도 판단.

## 1️⃣8️⃣ Semi-Supervised Learning
Labelling된 데이터와 비라벨 데이터 혼합 학습.  
- Labelling된 데이터를 우선 학습 → Pseudo Labelling 활용.

## 1️⃣9️⃣ Reinforcement Learning
Agent와 Environment의 상호작용을 통한 학습.  
- **Policy Function**: 특정 상태에서 행동 정의  
- **Reward/Value Function**: 특정 행동의 보상 정의.  
- 세부적으로 On/Off Policy Learning, Model-Based/Free Learning으로 구분.

## 2️⃣1️⃣ On-Policy Learning
행동하는 Agent와 학습하는 Agent가 같은 경우를 의미함.  
Agent가 직접 행동하면서 보상을 탐색하는 기본적인 알고리즘임.

## 2️⃣2️⃣ Off-Policy Learning
행동하는 Agent와 학습하는 Agent가 분리되어 있는 경우.  
- 행동 Agent: 기존 최적의 행동 수행  
- 학습 Agent: 새로운 보상을 탐색  
- 대표적인 알고리즘: **Q-Learning**

## 2️⃣3️⃣ Q-Learning
Agent가 특정 State에서 특정 행동을 통해 얻는 보상의 기대값(Q값)을 학습하는 알고리즘.  
- **Q Table**: 모든 상태와 행동의 보상을 저장  
- **Function Approximation**: 상태 공간이 큰 경우 근사치 사용  
- **Epsilon-Greedy**: Exploration과 Exploitation의 균형을 맞춤.

## 2️⃣4️⃣ Model-Based Learning
환경의 명시적인 정보를 포함한 모델을 사용하여 학습.  
- 행동을 직접 수행하지 않고도 보상을 예측 가능  
- **Planning**: 미래의 방향성을 계획  
- Dynamic Programming으로 구현 가능.

## 2️⃣5️⃣ Model-Free Learning
환경 모델 없이 Agent가 직접 행동하며 보상을 탐색.  
- Q-Learning이 대표적인 Model-Free 학습 방식.

## 2️⃣6️⃣ Discount Factor
미래 보상의 현재 가치 반영 정도를 결정하는 변수.  
- **높은 할인율**: 미래 가치 중시 → Exploration 초점  
- **낮은 할인율**: 현재 가치 중시 → Exploitation 초점.


## 2️⃣7️⃣ Multi-Armed Bandit
강화학습 알고리즘으로, 여러 슬롯머신 중 최고의 보상을 얻는 방법을 학습.



## 2️⃣8️⃣ Curse Of Dimension
차원의 저주는 데이터의 차원이 높아질수록 학습이 어려워지는 문제.  
- 해결 방법: **SVD**, **PCA** 등 차원 축소 기술 사용.



## 2️⃣9️⃣ Singular Value Decomposition (SVD)
행렬을 3개의 행렬(왼쪽 직교 행렬, 대각 행렬, 오른쪽 직교 행렬)로 분해하는 방법.  
- 기저벡터, 특이값, 열공간 정보를 포함.



## 3️⃣0️⃣ Principal Component Analysis (PCA)
데이터의 분산을 기준으로 차원을 축소하여 학습 효율을 높이는 기법.  
- 공분산 행렬 계산 → SVD로 특이값 행렬 구하기 → Eigenvector 선정.



## 3️⃣1️⃣ Gradient Vanishing
딥러닝 네트워크가 깊어지며 가중치 업데이트량이 0에 가까워지는 현상.  
- Sigmoid, Tanh와 같은 활성화 함수에서 발생.



## 3️⃣2️⃣ Gradient Exploding
역전파 과정에서 가중치 값이 과도하게 커지는 현상.  
- **Gradient Clipping**: 가중치 크기 제한으로 해결 가능.



## 3️⃣3️⃣ Backpropagation
Loss를 역방향으로 전파하여 가중치 업데이트를 수행하는 알고리즘.



## 3️⃣4️⃣ Variational AutoEncoder (VAE)
데이터 생성 모델로, 인코더와 디코더로 구성.  
- 인코더: Latent Space에 데이터 핵심 특징 저장  
- 디코더: Latent Space로 원본 데이터 복구.



## 3️⃣5️⃣ Latent Space
데이터의 핵심 특징을 압축하여 표현하는 공간.



## 3️⃣6️⃣ GAN (Generative Adversarial Network)
생성자와 판별자가 경쟁적으로 학습하여 데이터를 생성하는 모델.  
- **Mode Collapse**: 특정 패턴 데이터만 생성하는 문제.



## 3️⃣7️⃣ Attention Mechanism
데이터의 중요한 부분에 집중하여 학습 성능을 향상.  
- RNN 기반에서 발전하여 **Self-Attention**으로 Transformer 모델에 사용됨.



## 3️⃣8️⃣ Transfer Learning
사전 학습된 모델의 가중치를 새로운 학습에 적용.  
- **Fine Tuning**: 소규모 데이터셋에 미세 조정.



## 3️⃣9️⃣ Convolution Neural Network (CNN)
이미지 처리에 특화된 알고리즘.  
- **Convolution Layer**, **Pooling Layer**로 구성  
- YOLOv7과 같은 모델에 활용됨.



## 4️⃣0️⃣ Convolution Layer
작은 크기의 Filter(또는 Kernel)를 이미지 위에 슬라이딩하여 특징을 추출.



## 4️⃣0️⃣ Pooling Layer
풀링 계층은 Convolution을 통해 만들어진 Feature Map의 크기를 줄이는 역할을 수행.  
- **Max Pooling**: 사각 행렬의 최대값으로 축소  
- **Average Pooling**: 사각 행렬의 평균값으로 축소  
- 중요한 정보를 잘 유지하는 **Max Pooling**이 주로 사용됨.



## 4️⃣1️⃣ Flatten
행렬 내 값을 Weight에 곱하기 수월하도록 열벡터로 펼치는 연산.



## 4️⃣2️⃣ Padding
Convolution 연산 시 가장자리 특징 추출이 어려운 문제 해결.  
- 이미지 가장자리에 0 값을 추가하여 크기를 유지하면서 연산.



## 4️⃣3️⃣ Recurrent Neural Network (RNN)
시계열 데이터를 처리하는 모델.  
- 입력과 출력을 시퀀스 단위로 처리  
- **장기 의존성 소실 문제** 존재 → 이를 보완한 **LSTM**, **GRU** 아키텍처 제안.



## 4️⃣4️⃣ Long-Short Term Memory (LSTM)
RNN의 장기 의존성 소실 문제를 극복하기 위해 제안된 아키텍처.  
- 구성 요소: 기억 셀, 입력 게이트, 출력 게이트, 망각 게이트  
- 중요 정보는 유지하고, 중요하지 않은 정보는 제거.



## 4️⃣5️⃣ Gated Recurrent Unit (GRU)
LSTM의 단순화된 형태.  
- **업데이트 게이트**: 중요한 정보를 유지하고 새로운 정보를 받아들이는 역할  
- **리셋 게이트**: 중요하지 않은 정보를 잊는 역할.



## 4️⃣6️⃣ Hyperparameter
학습 시 사용자가 직접 설정하는 값.  
- 예: Epoch, Batch Size 등.



## 4️⃣7️⃣ Epoch
데이터셋을 몇 번 반복하여 학습할지 결정하는 Hyperparameter.



## 4️⃣8️⃣ Batch Size
한 번에 처리할 데이터 묶음의 크기.  
- **크게 설정**: 빠른 학습, 높은 메모리 사용  
- **작게 설정**: 느린 학습, 낮은 메모리 사용  
- **2의 배수**로 설정 권장.



## 4️⃣9️⃣ Grid Search
최적의 하이퍼파라미터 조합 탐색 기법.  
- 모든 조합 시도 → 연산량 증가.



## 5️⃣0️⃣ Random Search
하이퍼파라미터의 일부 조합만 랜덤하게 시도.  
- 연산량 감소, 국소 최적해에 빠질 가능성 존재.



## 5️⃣1️⃣ Parameter
학습 과정에서 모델이 동적으로 변경하는 값.  
- 예: Weight, Loss 등.



## 5️⃣2️⃣ Weight
모델 학습 과정에서 특정 요인에 영향을 결정하는 변수.



## 5️⃣3️⃣ Learning Rate
가중치 업데이트 크기를 결정.  
- 너무 크면 발산, 너무 작으면 느린 학습.



## 5️⃣4️⃣ Loss
예측값과 실제 값의 차이를 정의하는 함수.  
- **L1 Loss**, **L2 Loss**, **Cross Entropy Loss** 등이 있음.



## 5️⃣5️⃣ L1 Loss
예측값과 정답값의 차이를 **Mean Absolute Error**로 정의.  
- 계산식: \( \frac{1}{N} \sum_{i=1}^{N} |Y_{\text{pred}} - Y_{\text{true}}| \)



## 5️⃣6️⃣ L2 Loss
예측값과 정답값의 차이를 **Mean Squared Error**로 정의.  
- 계산식: \( \frac{1}{N} \sum_{i=1}^{N} (Y_{\text{pred}} - Y_{\text{true}})^2 \)



## 5️⃣7️⃣ Explainable Artificial Intelligence (XAI)
AI가 예측한 결과의 근거를 설명할 수 있는 모델.  
- 예: 어떤 요인이 얼마나 영향을 미쳤는지 해석 가능.



## 5️⃣8️⃣ Overfitting
모델이 학습 데이터에 지나치게 적응하여 실제 데이터 예측 성능이 저하되는 현상.  
- 해결책: Cross-Validation, Early-Stopping, Dropout, Regularization 등.



## 5️⃣9️⃣ Model Complexity
모델의 복잡도를 의미.  
- **적당한 복잡도**: 좋은 모델  
- 너무 복잡하거나 단순한 모델은 일반화 성능 저하.



## 6️⃣1️⃣ Bias
편향은 모델이 얼마나 단순하여 복잡한 패턴을 학습하지 못하는가를 의미.  
- **편향이 높을수록**: 모델이 단순하여 복잡한 패턴 학습 어려움.



## 6️⃣2️⃣ Variance
모델이 데이터의 국소적 특징에 얼마나 민감한지를 의미.  
- **분산이 높을수록**: 작은 특징에도 과도하게 민감하게 반응.



## 6️⃣3️⃣ Bias-Variance Tradeoff
편향과 분산은 서로 상충 관계에 있음.  
- 편향 ↑ → 분산 ↓  
- 편향 ↓ → 분산 ↑  
- 적절한 균형 필요.



## 6️⃣4️⃣ Generalization Ability
모델이 학습 데이터뿐 아니라 실제 데이터에 대한 예측을 잘 수행하는 능력.



## 6️⃣5️⃣ Cross Validation
Train Data 일부를 Test Data로 사용하여 두 데이터에 대한 Loss를 동시에 추적.



## 6️⃣6️⃣ K-Fold Cross Validation
Train Data를 K개의 서브셋으로 나누어 각 서브셋을 Test Data로 사용하여 교차 검증 수행.



## 6️⃣7️⃣ Dropout
Overfitting 방지를 위해 학습 과정에서 일부 뉴런을 랜덤하게 비활성화.



## 6️⃣8️⃣ Early-Stopping
학습이 더 이상 개선되지 않을 때 조기 종료.  
- Train Data와 Test Data의 Loss를 동시에 추적하여 설정.



## 6️⃣9️⃣ Regularization
모델 성능 개선을 위해 손실 함수에 가중치 크기를 함께 고려.  
- **L1 Regularization**: 필요 없는 가중치를 0으로 만듦 (Lasso).  
- **L2 Regularization**: 큰 가중치 값을 줄임 (Ridge).



## 7️⃣0️⃣ Batch Normalization
각 미니배치의 입력 데이터를 표준화하여 Internal Covariance Shift를 줄임.  
- 입력 데이터 평균을 0, 분산을 1로 만듦.



## 7️⃣1️⃣ Data Augmentation
데이터를 복제하고 Masking, Resizing, Rotation 등을 적용하여 데이터셋 양을 증가.



## 7️⃣2️⃣ Underfitting
모델이 편향이 높아 학습 데이터조차 잘 예측하지 못하는 현상.  
- Epoch 수를 늘리거나 모델 복잡도를 증가시켜 해결 가능.



## 7️⃣3️⃣ Data Imbalance
특정 클래스 데이터가 다른 클래스에 비해 과도하게 적은 현상.  
- **해결 방법**:  
  - 언더샘플링  
  - 오버샘플링  
  - 가중치 조정.



## 7️⃣4️⃣ Decision Tree
계층형 구조로 데이터를 Leaf Node에 도달할 때까지 분류하는 알고리즘.  
- **데이터 배치 기준**: Gini Impurity, Entropy.



## 7️⃣5️⃣ Gini Impurity
여러 클래스가 얼마나 섞여 있는지 측정.  
- 값: \( 1 - \sum_{i=1}^{C} P_i^2 \)



## 7️⃣6️⃣ Entropy
클래스 섞임으로 인해 예측이 얼마나 어려운지 측정.  
- 값: \( -\sum_{i=1}^{C} P_i \log_2 P_i \)



## 7️⃣7️⃣ Random Forest
여러 Decision Tree를 Bagging으로 결합하여 Overfitting 위험을 줄이고 예측력 향상.



## 7️⃣8️⃣ Ensemble Learning
여러 모델을 결합하여 학습.  
- **방법**: Bagging, Boosting, Stacking.



## 7️⃣9️⃣ Bagging
데이터를 여러 서브셋으로 나누고 각 서브셋을 독립적으로 학습.  
- Random Forest가 대표적.



## 8️⃣0️⃣ Boosting
순차적으로 학습하며 이전 모델의 단점을 보완하는 기법.  
- AdaBoost가 대표적.



## 8️⃣1️⃣ Stacking
서로 다른 모델의 예측 결과를 Meta Model이 조합하여 최종 예측.



## 8️⃣2️⃣ Natural Language Processing (NLP)
문자 데이터를 저차원의 숫자 벡터로 변환하여 이해하거나 번역, 생성에 활용하는 기술.



## 8️⃣6️⃣ Word2Vec
문자 데이터를 저차원 숫자 벡터로 변환하여 컴퓨터가 의미와 관계를 이해할 수 있게 하는 Word Embedding 기법.  
- **학습 핵심**: 관계있는 단어는 가깝게, 관계없는 단어는 멀리 배치.



## 8️⃣7️⃣ Skip-Gram
중심 단어로 주변 단어를 예측하는 기법.  
- 희귀한 단어 예측에 특화.



## 8️⃣8️⃣ Continuous Bag Of Words (CBOW)
주변 단어로 중심 단어를 예측하는 기법.  
- 일반적인 단어 예측에 특화.



## 8️⃣9️⃣ Tokenization
문장을 작은 단위(Token)로 분리하는 작업.  
- 예: **저는 지원자 송현교입니다** → '저는', '지원자', '송현교', '입니다'



## 9️⃣0️⃣ Stemming
단어에서 접미사를 제거하여 기본형을 추출하는 방법.  
- 빠르지만 낮은 정확도를 보임.  
- 예: **Learned, Learning** → **Learn, Learn**



## 9️⃣1️⃣ Lemmatization
품사의 의미까지 분석하여 정확한 기본형을 추출하는 방법.  
- 느리지만 Stemming보다 정확.  
- 예: **Studied, Studies** → **Study, Study**



## 9️⃣2️⃣ Morphological Analysis
문장을 형태소(Token의 더 작은 단위)로 분리.  
- 예: **저는 지원자 송현교입니다** → '저', '는', '지원', '자', '송현교', '입니다'



## 9️⃣3️⃣ Representation Learning
데이터를 저차원 숫자 벡터로 표현하며, 유의미한 패턴을 추출 및 학습.



## 9️⃣4️⃣ Seq2Seq
입력 시퀀스를 압축 및 해제하여 처리하는 자연어 모델.  
- **구성**: 인코더와 디코더.  
- 예: 문장 번역, 문장 생성.



## 9️⃣5️⃣ Topic Modeling
문서의 단어 등장 빈도와 관계를 기반으로 문서의 주제를 분석.



## 9️⃣6️⃣ Transformer
Attention Mechanism을 기반으로 NLP 성능을 혁신한 모델.  
- **Self-Attention Mechanism**: 데이터의 중요한 부분에 집중하며 RNN의 한계를 극복.



## 9️⃣7️⃣ Bidirectional Encoder Representation from Transformers (BERT)
구글에서 개발한 NLP 모델.  
- 기존 순방향 처리와 달리 **양방향 처리**로 높은 이해도와 부드러운 문장 구조 생성.



## 9️⃣8️⃣ Confusion Matrix
분류 모델 성능 평가 지표를 2x2 행렬로 나타낸 것.  
- **TP**: True Positive  
- **TN**: True Negative  
- **FP**: False Positive (Type 1 Error)  
- **FN**: False Negative (Type 2 Error)  

### 주요 계산식
- **Recall (재현율)**: \( \text{TP} / (\text{TP} + \text{FN}) \)  
- **Precision (정밀도)**: \( \text{TP} / (\text{TP} + \text{FP}) \)  
- **F1 Score**: \( 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)



## 9️⃣9️⃣ Receiver Operating Characteristic Curve (ROC Curve)
분류 모델 성능을 평가하는 곡선.  
- **x축**: FP Rate  
- **y축**: TP Rate  
- 그래프가 좌측 상단에 가까울수록 좋은 모델.



## 🔟0️⃣ Area Under the Curve (AUC)
ROC Curve의 하단 면적.  
- **1**: 완벽한 분류  
- **0.5**: 랜덤한 분류  
- **0.5 이하**: 잘못된 분류.



## 9️⃣7️⃣ Recommender System
추천 시스템은 Collaborative Filtering 기반으로, 사용자가 관심 가질 만한 Item을 추천하는 기술.



## 9️⃣8️⃣ Collaborative Filtering
유사한 취향이나 취미를 가진 사용자들이 동일한 Item을 선호한다는 원리에 기반한 필터링 시스템.



## 9️⃣9️⃣ (Batch) Gradient Descent
손실함수를 최적화하기 위한 알고리즘.  
- **원리**: 손실함수를 다변수 함수로 가정, 각 변수에 대해 편미분 수행 → Gradient의 반대 방향으로 이동.  
- **단점**: 모든 데이터 포인트를 사용해 Gradient 계산 → 연산량 많고 느림.  
- 해결: **SGD, Mini-Batch GD, Momentum, Adam** 등 제안.



## 🔟0️⃣ Stochastic Gradient Descent (SGD)
하나의 데이터 포인트에 대해 Gradient를 계산하고 업데이트.  
- 빠르지만 노이즈에 민감.

---

## 🔟1️⃣ Mini-Batch Gradient Descent
Batch GD와 SGD의 절충안으로, Mini-Batch 단위로 Gradient 계산 및 업데이트.  
- 빠르고 안정적 수렴.



## 🔟2️⃣ Momentum
이전 가중치 업데이트량을 활용하여 빠르게 수렴하도록 가속.



## 🔟3️⃣ Nesterov Accelerated Gradient (NAG)
Momentum 기반으로 가중치 업데이트 방향을 예측하고, 예측 위치에서 기울기 계산.  
- Momentum보다 빠름.



## 🔟4️⃣ Adagrad
자주 등장하는 파라미터는 작은 학습률, 드물게 등장하는 파라미터는 높은 학습률 적용.



## 🔟5️⃣ RmsProp
지수 가중 이동 평균을 사용해 최근 Gradient에 높은 학습률을 적용하고 과거 Gradient는 잊음.



## 🔟6️⃣ Adam
Gradient의 1차 Moment(평균)와 2차 Moment(분산)을 추적하여 가중치 업데이트.  
- **Adagrad와 RmsProp 결합**.



## 🔟7️⃣ Density Estimation
확률밀도함수를 구하기 위해 요소의 분포를 분석하는 기법.  
- **모수적 측정**: Gaussian Distribution  
- **비모수적 측정**: 히스토그램.



## 🔟8️⃣ Feedback Loop
출력이 다시 입력으로 피드백되는 순환적 구조.



## 🔟9️⃣ Depthwise Separable Convolution
RGB 채널을 개별적으로 처리한 후 결합하는 방식으로, 효율적인 합성곱 연산.  
- 사용 사례: MobileNet, EfficientNet, YOLOv7.



## 🔟0️⃣ Depthwise Convolution
RGB 채널을 분리, 각각 필터를 적용해 개별 특징 맵 생성.



## 🔟1️⃣ Pointwise Convolution
1x1 Convolution으로 Depthwise Convolution의 특징 맵을 결합.



## 🔟2️⃣ KL Divergence
두 확률분포 간의 거리를 계산하여 정보 손실량을 측정.



## 🔟3️⃣ Central Limit Theorem
표본평균의 분포가 표본 크기 \(N\)이 충분히 크다면 Gaussian Distribution을 따르게 되는 이론.  
- 계산식: \( \frac{1}{N} \sum_{i=1}^{N} X_i \).



## ⭐ YOLOv7 ⭐
YOLOv7은 CNN 기반 실시간 객체 탐지 모델로, 필자가 캡스톤디자인 프로젝트에 사용한 모델입니다.  
Mask R-CNN 대비 **550% 빠른 속도**와 **10% 높은 정확도**를 보여줍니다.  

### 핵심 기술
1. **Extended Efficient Label Aggregation Network**:  
   공간 정보와 특징을 효율적으로 추출 및 결합.

2. **Model Reparameterization**:  
   학습 중 복잡한 레이어를 학습 완료 후 단순화하여 경량화.

3. **Dynamic Label Aggregation**:  
   다양한 크기의 Anchor Box로 객체 탐지 및 최적 바운딩 박스 출력.

4. **Bag Of Freebies**:  
   데이터 증강으로 사용자 전처리 없이도 일반화 능력 향상.

5. **ConvNexT**:  
   CNN의 발전 형태.  
   - **Layer Normalization**: Batch Normalization 대신 사용 → 시퀀스 데이터에 적합.  
   - **Gaussian Error Linear Unit**: ReLU 대체로 음수 값 부드럽게 처리.  
   - **Depthwise Convolution**: 효율적인 연산 수행.  
   - **ResNet의 Skip-Connection**: 잔차 연결로 정보 손실 방지.
