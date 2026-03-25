# Project-01.-No-Show
병원 노쇼원인 분석 및 해결방법 고안
## **[Project Title]** 
 “ML 기반 병원 No-show 원인 분석 및 예약 리텐션 개선 전략"

## **[Summary]**
- **배경:** 예약 노쇼(No-show)로 인한 의료 자원 낭비 및 병원 손실 최소화
- **데이터:** 약 10만 건의 환자 방문 기록 (Kaggle Dataset)
- **사용 툴:** Python (Numpy, Pandas, Matplot…)

## **[Data Preprocessing]**
**1. 노이즈 제거 및 데이터 정제**

- **노이즈 제거 및 데이터 정제**
    - **불필요한 식별자 제거:** 분석에 유의미하지 않은 PatientId, AppointmentID 삭제
    - **논리적 오류 수정:** 예약일보다 진료일이 앞서는 데이터 오류 (Date.diff 음수값) 제거
    - **데이터 타입 최적화:** 모든 bool형, object형 변수를 int형로 변환하여 모델학습에 최적화
    - **적은 표본의 데이터 필터링:** Neighborhood의 표본수가 적은 지역 제거

**2. 피처 통합**
- **질병 지수(chronic_score) 생성**
    - Hipertension, Diabetes, Alcoholism(개별 질병 칼럼)을 합산하여 ‘환자 기저질환 중증도’를 나타내는 단일 지표 생성.
    - 다중신공성 문제 해결 및 모델 해석력 증가를 위함. EDA이후 개별 칼럼은 삭제

**3. 파생변수 생성**
- **지역별 노쇼 위험도 (region_noshow_rate)**
    - 각 지역의 평균 노쇼 비율을 계산하여 매핑.
    - 지역적 특성이 노쇼에 미치는 영향을 확인하기 위함
    - neighborhood는 이후 모델 test/train 데이터를 만들 때 데이터 누수 발생을 막기 위해 다시 region_noshow_rate 파생변수를 생성 후 삭제할 예정
 
## **[EDA]**
**1. 숫자형 변수 상관관계 확인** 

- **Heat Map**
  
    - 의미있는 상관관계
      
        Hipertension ↔ chronic_score (0.86)

        Diabetes ↔ chronic_score (0.71)

        Alcoholism ↔ chronic_score (0.34)

        Handcap ↔ chronic_score (0.3)
      
        ⇒ 질병 칼럼과 choronic_score간 상관관계: choronic_score만으로도 질병 칼럼의 정보를 얻을 수 있기 때문에 추후 모델링 feature에서 제외 

**2. Date.diff와 No_show**

- Date.diff에 따른 No_show 비율
  
  → 예약 대기시간이 길수록 No_show 증가함을 확인
  
- Date.diff에 따른 No_show 비율 구간별 추가 확인
  
  → 대기 기간이 증가할수록 No-show 비율이 증가하는 경향이 나타남.
  
  → 특히 당일 예약과  대기 기간 30일 이상인 경우를 비교했을 때, 약 28% 수준의 차이를 보임.  

**3. SMS_received와 No_show**

- SMS_recieved 따른 No_show 비율
  → SMS를 수신했을 때 No_show 비율이 증가
  
  **confounding bias**: SMS 수신 여부에 따른 노쇼율 차이가 환자의 특성 차이에서 발생한 것인지 확인하기 위해 age와 chronic_score, region_noshow_rate를 추가 변수로 고려

   - **Age와 No_show**
 
     - Age 그룹별 특징 : 환자 비율 Young Adult, Middle Age 그룹이 가장 높음, 나이가 많을수록 가진 질병의 개수 평균이 높음
  
     - Age 그룹별 No_show 비율 : child를 제외한 나이대에서 나이가 어릴수록 No-show 비율이 높음
  
     - Age 그룹 별 SMS_recieved 비율 : 나이대별 SMS 수신 비율 차이 미비, 환자 비중이 높은 연령대에서 No-show 비율이 상대적으로 높게 나타남

       → 이러한 연령 분포가 SMS_received에 반영되면서 SMS 수신 그룹에서 No_show 비율에 편향이 발생한 것으로 해석됨

       → **SMS 수신 여부의 영향보다는 나이의 영향이 큼**

  - **choronic_score과 No_show**
  
      - choronic_score에 따른 No_show 비율
  
         → choronic_score = 4일 때 수치가 높으나 데이터가 적어 생긴 오류로 확인 (Distribution of Choronic Disease Score)
  
         → choronic_score가 높은 환자일수록 No_show 비율이 감소
  
         → choronic_score = 0인 환자가 우세한 가운데, 이 그룹의 No_show율이 가장 높음. 이 분포가 SMS 수신 여부 변수에 반영되면서 SMS 수신 그룹에서 노쇼 비율이 높게 나타나는 **편향(confounding effect)** 이 발생한 것으로 해석됨.
  
         → **SMS 수신 여부의 영향보다는 질병 개수의 영향이 큼**
  
  - **region_noshow_rate와 No_show**
  
    - No_show비율 상위 3개, 하위 3개 지역 비교
  
      → 상위 지역과 하위 지역 간 노쇼율 차이는 약 2배 수준(약 14%p)으로 나타남.
  
      → 이는 노쇼 발생이 개인의 지역 환경의 영향을 받을 수 있음을 나타냄.
  
    - 추가 변수를 확인 결과, SMS 수신 여부에 따른 No_show비율은 환자 특성 차이에 의한 왜곡된 결과일 가능성이 높음. 정확한 평가를 위해 모델링을 통한 추가 분석이 필요
  
**4. Scholarship과 No_show**

- Scholarship에 따른 No_show 비율

    → 지원 대상 환자 No_show 비율이 높음. 경제적 부담이 적어 예약에 대한 낮은 책임감으로 인한 결과로 해석할 수 있다.

## **[머신러닝]**

- **1. Feature Engineering**
  1. **Chronic Disease Count:** 만성 질환이 많으면 병원 방문 확률이 높아지는 결과 하에 대입
  2. **질병 유무 추가 (Has Chronic Disease):** 모델의 학습을 돕기 위해 Binary feature 추가
  3. **High Risk Wating Flag:** 14일 이후 no-show가 급증하는 패턴을 반영하여 대기시간이 길어지는 경우 따로 Flag
  4. **SMS + Long Wait Interaction:** 대기시간이 긴 환자에게 SMS 수신이 효과가 있는지 모델이 학습
    
- **2. Model - Logistic Regression**
  
    → Logistic Regression은 이진 분류 문제에서 가장 기본적인 Baseline 모델로, 각 변수의 영향 방향을 해석하기 쉽다는 장점이 있어 선택
  
    - **결과 해석**
      
        - 가장 영향이 큰 변수: **Date.diff** (대기시간이 길수록 노쇼 확률 증가)
        - **SMS_reeived:** SMS를 받을수록 No-show 비율 증가
        - **region_noshow_rate:** No_show 많은 지역에서 No_show 비율 증가
            → Feature가 제데로 반영되었음을 알 수 있음
        - **Age:** 나이가 많을수록 No_show 비율 감소
        - **sms_long_wait:** 긴 대시시간 환자에게 SMS가 가면 No_show 비율 감소
        
- **3. Model - Random Forest**
    - Logistic Regression에서 발견하지 못한 비선형 관계를 추가로 확인하고자 비선형 Baseline 모델로 Random Forest 선택
    - Feature Importance는 각 변수가 모델의 예측 성능 향상에 기여한 정도를 나타냄. 값이 높을수록 No_show 여부를 설명하는 데 중요한 변수임을 의미.
        
    - **결과 해석**
        - 가장 영향이 큰 변수: **Date.diff (**예약 대기시간이 길수록 No-show 확률 증가)
        - **Age:** 나이가 많을수록 No-show 비율 감소
        - **region_noshow_rate:** 노쇼가 많은 지역일수록 No-show 확률 증가
  
