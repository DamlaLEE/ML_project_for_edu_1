# 🧠 Student Performance ML Project

(가상의 시나리오) '교육 서비스 A' 에서 학생의 점심, 학부모의 교육 수준, 인종 등을 고려하여 학생의 **향후 평균 학습 성적을 예측**하는 머신러닝 회귀 모델을 구현했습니다.  
이를 통해 서비스 등록 시 필요한 절차를 자동화하거나, 개인화된 학습 경로를 제시하는 데 활용할 수 있습니다.

---
## 📂 프로젝트 구조

```
ML_project_for_edu1/
│
├── utils/
│   └── utils_group.py          # ML 관련 함수 모음
│
├── main.py                     # 전체 ML 모델 학습 및 실행 코드
├── StudentsPerformance.csv     # 원본 데이터
├── basic info.txt              # 기본 정보 정리
└── README.md                   # 프로젝트 소개 파일
```

## 📚 사용된 주요 라이브러리

- **기본** : `os`, `pickle`
- **데이터 전처리 & 분석** : `pandas`, `numpy`
- **시각화** : `seaborn`, `matplotlib`
- **머신러닝** :
  - 모델링 : `scikit-learn`, `xgboost`, `lightgbm`
  - 하이퍼파라미터 탐색 : `GridSearchCV`, `RandomizedSearchCV`
  - 통계 : `scipy.stats`
 
---

### 📄 3.2 데이터 설명

데이터는 학생들의 성별, 점심 급식 상태, 시험 준비 여부, 학부모의 교육 수준 등의 특성으로 구성되어 있으며, 아래와 같은 방식으로 정제되었습니다.

| 변수명                       | 값           | 설명                                      |
|----------------------------|--------------|-------------------------------------------|
| **gender**                 | 0            | 남성 (male)                               |
|                            | 1            | 여성 (female)                             |
| **lunch**                  | 0            | free/reduced (무상 또는 축소형 급식)     |
|                            | 1            | standard (표준형 급식)                   |
| **test preparation course**| 0            | 시험 준비 과정 미참여                    |
|                            | 1            | 시험 준비 과정 완료                      |
| **parental level of education** | 1~6     | 학부모의 교육 수준 (아래 참조)           |
| **score_average**          | mean()       | 3과목 점수의 평균 (maths, reading, writing) |

**parental level of education 인코딩 기준**:

| 학력 구분               | 인코딩 값 |
|------------------------|-----------|
| some high school       | 1         |
| high school            | 2         |
| some college           | 3         |
| associate's degree     | 4         |
| bachelor's degree      | 5         |
| master's degree        | 6         |

---

## 1️⃣ 문제 상황

> A 교육 서비스에서 **학생의 점심 여부, 학부모의 교육 수준, 인종 등의 정보**를 바탕으로  
> 학생의 향후 학습 수준(평균 점수)을 예측하고,  
> 그 결과를 기반으로 **서비스 등록에 필요한 개인화 절차**를 자동화하고자 합니다.

---

## 2️⃣ 해결 방안

- 학생의 **3과목 점수(maths, reading, writing)** 를 기반으로 평균 점수를 만들고
- 다양한 회귀 기반 머신러닝 모델을 적용하여
- 최종적으로 가장 예측 성능이 우수한 Ridge 모델을 **챔피언 모델**로 선정하여 저장했습니다.

## 📦 프로젝트 확장 가능성

- 실시간 학생 성적 데이터 입력 시, 자동으로 예상 평균 점수 제공
- 교육 커리큘럼 추천, 학습 리스크 분석, 성적 개선 가이드 등에 활용 가능
- 모델 배포를 통해 서비스 내 추천 시스템으로 연동 가능

---

## 3️⃣ 상세 코드 진행 방식

### ✅ Step 1 : 데이터 로딩 + EDA
- 데이터를 불러온 후 기본적인 형태, 결측치, 분포 등을 확인
- 주요 변수 간의 상관관계를 시각화
  - `parental level of education` 과 `score_average`
  - `test preparation course` 과 `score_average`

### ✅ Step 2 : 머신러닝 학습
- **2.1 데이터 분할**  
  - 학습용 800개 / 테스트용 200개로 분할  
- **2.2 다양한 ML 모델 학습**
  - 선형 모델: Linear, Ridge, Lasso
  - 트리 모델: RandomForest, XGBoost, LightGBM
- **2.3 Feature Importance 확인**
  - 트리 모델 기반으로 중요 피처 확인
- **2.4 Cross Validation**
  - Stratified K-Fold로 모델별 안정성 확인
- **2.5 하이퍼파라미터 튜닝**
  - Ridge, Lasso 모델 대상으로 GridSearchCV 수행

### ✅ Step 3 : 챔피언 모델 선정 및 테스트 예측
- 가장 성능이 좋은 모델로 테스트 데이터 예측
- 예측 결과 저장 및 평가

---

## 4️⃣ 예측 성능 지표

머신러닝 회귀 평가 지표를 사용하여 성능을 평가합니다.

| 지표명 | 설명 |
|--------|------|
| **RMSE** | Root Mean Squared Error (평균 제곱근 오차) |
| **MAE**  | Mean Absolute Error (평균 절대 오차) |
| **Accuracy** | 예측값 대비 실제값의 오차가 5% 이하인 비율<br>(정확도 지표가 아님!)<br>계산식: `((abs(y_pred - y_train) / y_train) < 0.05).mean()` |

---

## 🧪 머신러닝 학습 차수별 요약

### 🔹 1차 학습
- 선형(3종), 트리 모델(3종) 총 6개 비교
- 트리 모델이 학습 성능은 높았지만, 테스트 데이터에서는 성능이 하락 → 과적합 가능성

#### ✅ 1차 학습 결과 (기본 모델 성능 비교)

| Model | RMSE (Train) | RMSE (Test) | MAE (Train) | MAE (Test) | Accuracy (Train) | Accuracy (Test) |
|-------|--------------|-------------|-------------|------------|------------------|-----------------|
| **Ridge** | 12.3373 | 12.8639 | 9.9407 | 10.4065 | 0.21125 | 0.195 |

### 🔹 2차 학습
- 트리 모델의 Feature Importance를 기반으로 Feature Engineering 시도
- 3개 모델의 중요도가 서로 달라 적용 보류

### 🔹 3차 학습
- Cross Validation (KFold) 적용
- 선형 모델이 높은 평균 성능과 낮은 표준편차로 안정성 우수
- → 선형 모델 기반으로 최종 모델 선정

#### ✅ 3차 학습 결과 (Cross Validation 적용)

> Cross Validation 결과 (cv=5)

| Model | Mean RMSE | Std |
|--------|------------|------|
| **Ridge** | 12.483 | 0.781 |

### 🔹 4차 학습
- `GridSearchCV`로 Ridge / Lasso 하이퍼파라미터 튜닝
- 최종적으로 Ridge 모델 (`alpha=10`) 이 성능 최우수

---

## 🏆 최종 챔피언 모델

- **선택 모델**: `Ridge` Regression  
- **최적 하이퍼파라미터**: `alpha = 10`
- **예측 성능**:

| RMSE (Test) | MAE (Test) | Accuracy (Test) |
|-------------|------------|-----------------|
| **12.802** | **10.487** | **0.2025** |

- 모델 저장 파일: `final_ridge_model.pkl` (pickle 사용)

---
## 📌 향후 ML 서빙 방식

- 실 서비스 적용 시 학습된 모델을 `.pkl` 파일로 저장하여 API 또는 배치 예측 시스템에 연결 가능
- 실제 학생의 성적 또는 등록 데이터가 입력되면, 예측 성적으로 학습지원 대상 또는 관심군 분류 가능

---

## 📎 부록

- **작성자**: DS_Yujin LEE  
- **작성일**: 2025-04-09 ~ 2025-04-10  
- **버전**: ver1  
- **데이터 출처**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
