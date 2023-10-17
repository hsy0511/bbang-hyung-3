# bbang-hyung-3

# 3강 회귀
### 목차
- 회귀 Regressor 소개
- 선형 회귀 Linear Regression 실습
## 분류와 회귀
### 강아지와 고양이 구분
강아지와 고양이의 중간은 없다.

### 부동산 가격 예측
부동산 가격은 연속적인 실수로 나타낼 수 있다.
- 1000만원
- 1001만원
- 1억 2500만원
- 51억 4123만원
## 선형회귀 실습
### 보스턴 집 값 데이터셋을 이용한 실습
#### 데이터셋 로드
- CRIM: 범죄율
- ZN: 25,000평방 피트 당 주거용 토지의 비율
- INDUS: 비소매 비즈니스 면적 비율
- CHAS: 찰스 강 더미 변수 (통로가 하천을 향하면 1; 그렇지 않으면 0)
- NOX: 산화 질소 농도 (1000만 분의 1)
- RM: 평균 방의 개수
- AGE: 1940년 이전에 건축된 자가 소유 점유 비율
- DIS: 5개의 보스턴 고용 센터까지의 가중 거리
- RAD: 고속도로 접근성 지수
- TAX: 10,000달러 당 전체 가치 재산 세율
- PTRATIO 도시별 학생-교사 비율
- B: 1000 (Bk-0.63) ^ 2 (Bk는 도시별 검정 비율)
- LSTAT: 인구의 낮은 지위
- target: 자가 주택의 중앙값 (1,000달러 단위)

```python
from sklearn.datasets import load_boston
import pandas as pd
// load_boston 데이터셋 패키지를 가져온다.
// 판다스 라이브러리를 가져온다.

data = load_boston()
// data에 load_boston를 데이터셋해준다.

df = pd.DataFrame(data['data'], columns=data['feature_names'])
// 데이터 프레임에 데이터 값은 load_boston을 데이터셋한 data를 넣어주고, 컬럼은 data의 feature_names로 넣어준다.

df['target'] = data['target']
// 데이터 프레임에 타겟 데이터는 data의 타겟 데이터로 넣어준다.

df.tail()
// 하위 5개의 데이터를 뽑아낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/78b583c1-2847-4a19-955d-8c0e40e0a7a0)

#### 데이터 시각화
- Distribution plot(분포 차트)

```python
import matplotlib.pyplot as plt
import seaborn as sns
// matplotlib, drsborn 라이브러리를 가져온다.

sns.displot(x=df['target'])
// x축이 타겟 값인 분포차트를 그린다

plt.show()
차트를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/a87fe7de-76ec-405e-8082-dcb302ad0290)

- correlation matrix(상관 행렬)

```python
plt.figure(figsize=(10, 10))
// 가로10 세로10인 차트를 생성한다.

corr = df.corr()
// 누락값을 제외하고 전체 데이터에 상관도를 계산함
sns.heatmap(corr, annot=True, square=True, cmap='PiYG', vmin=-1, vmax=1)
// annot : 수치표시
// square : 틀을 정사각형으로 지정 (false 지정시 직사각형)
// camp : 색상변경
// vmin, vmax : 최대 최소 기준점 변경

plt.show()
차트를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/dd74eb7f-9f3b-4fbc-8cbc-c4b14c2e512f)

#### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=2021)
// train_test_split를 사용하여 x에는 정규화된 데이터를 넣어주고 y에는 타겟을 넣어주고 val은 20%, train은 80%로 랜덤으로 나눈다.

print(x_train.shape, y_train.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.

print(x_val.shape, y_val.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/993287b4-69ea-433a-a7c4-0a802aabc1a8)

#### 모델 정의
```python
from sklearn.linear_model import LinearRegression
// LinearRegression 모델을 가져온다

model = LinearRegression()
// LinearRegression 모델을 정의한다.
```
#### 학습
```python
model.fit(x_train, y_train)
// 데이터를 학습시킨다.
```
#### 검증
```python
x_val[:5]
// x_val 배열에 0번째부터 4번째까지 뽑는다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/20957ee3-d857-40d7-8166-6dd5f43520d4)

```python
y_pred = model.predict(x_val)
// x_val 데이터 값을 예측한다.

print(y_pred[:5])
// 에측한 값을 0부터 4까지 뽑는다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/1b945e6d-d6ac-41c7-843f-3a1dc488be1c)

#### 회귀에서의 정확도는 어떻게 구할 수 있을까?
- 정답값과 예측값이 차이가 작으면 정확도가 높다
- 정답값과 예측값의 차이가 크면 정확도가 낮다

```python
print(list(y_val[:5]))
// y_val을 리스트형태로 0부터 4까지 출력한다.

print(list(y_pred[:5]))
// y_pred를 리스트형태로 0부터 4까지 출력한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6d1602bb-3e9e-461a-8043-fcaacf923cd6)

```python
y_val[:5] - y_pred[:5]
// y_val 0~4와 y_pred 0~4를 뺀다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b7b97f4e-bb7e-43fa-b3be-b7faf1dbe941)

```python
abs(y_val[:5] - y_pred[:5])
// y_val 0~4와 y_pred 0~4를 빼고 절대값으로 나타낸다.
// abs : 절대 값
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/e9322218-b204-44a9-9fc6-03a65ce4f9cd)

#### Mean Absolute Error
- MAE
- 정답과 예측값 차이의 절대값의 평균
- 정확도를 알기위해서(작을수록 정확도가 높다)

```python
abs(y_val[:5] - y_pred[:5]).mean()
// 0~4까지 절대값의 평균을 나타낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/3a4d443b-f7e3-4f55-9b44-893f6b09ed81)

```python
abs(y_val - y_pred).mean()
// 전체 데이터 절대값의 평균을 나타낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/7988b70b-9892-4fd8-bf3f-59c91c4ef58a)

```python
from sklearn.metrics import mean_absolute_error
// mean_absolute_error 절대값의 차이의 평균 패키지를 가져온다.

mean_absolute_error(y_val, y_pred)
// 정답값과 예측값의 차이의 평균.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/9acba601-5c6d-4068-a9c6-2948058749c3)

#### Mean Squared Error
- MSE
- 정답과 예측값 차이의 제곱의 평균
- 제곱으로 양수를 만들기 위해서
- 제곱을해서 에러를 크게 만들어서 정확도를 더 좋은 모델을 만들때 사용한다.  

```python
((y_val - y_pred) ** 2).mean()
// 절대값의 차이의 제곱의 평균
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b36b829f-11b1-4eb1-aa06-ff12fd7ac77b)

```python
from sklearn.metrics import mean_squared_error
// mean_squared_error 절대값의 차이의 제곱의 평균 패키지를 가져온다.

mean_squared_error(y_val, y_pred)
// 정답값과 예측값의 차이의 제곱의 평균.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/bbb445e0-b02f-43d8-b387-00c17a5a2743)

#### 표준화
```python
from sklearn.preprocessing import StandardScaler
// StandardScaler 표준화 패키지를 가져온다

scaler = StandardScaler()
// StandardScaler() 객체 생성

x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)
// x_train, x_val 데이터 값을 표준화시켜서 x_train_scaled, x_val_scaled 변수에 저장한다.

print(x_train_scaled[:5])
// x_train_scaled 0~4까지 출력한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/98c6f379-4e65-4fd3-a7a2-48516aabfb3a)

```python
model = LinearRegression()
// 모델 정의

model.fit(x_train_scaled, y_train)
// 모델 훈련

y_pred = model.predict(x_val_scaled)
// 정답값 예측

mean_absolute_error(y_val, y_pred)
// 예측한 값의 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/fcf51783-e3e1-4912-9ead-a6dd2992d4b7)

#### 정규화
```python
from sklearn.preprocessing import MinMaxScaler
// MinMaxScaler 정규화 패키지를 가져온다

scaler = MinMaxScaler()
// MinMaxScaler() 객체를 생성한다.

x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.fit_transform(x_val)
// x_train, x_val 데이터 값을 정규화시켜서 x_train_scaled, x_val_scaled 변수에 저장한다.

print(x_train_scaled[:5])
// x_train_scaled 0~4까지 출력한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/afd255f7-f357-4083-aff6-eacf52a5ed7f)

```python
model = LinearRegression()
// 모델 정의

model.fit(x_train_scaled, y_train)
// 모델 훈련

y_pred = model.predict(x_val_scaled)
// 정답값 예측

mean_absolute_error(y_val, y_pred)
// 예측한 값의 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/3ae7ebb8-5ce6-40ae-8fa9-fb5ad9f2d737)

### Linear Regression (선형회귀)
![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/399937d7-e4ae-4e42-9a10-acbca1922e0e)

분류에서는 선이 svm(support vector muchine)으로 svm 기준으로 분류가 되는 것 이지만

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5504e1e9-b970-4864-be14-16812bdd2e12)

선형회귀는 선이 정답값, 예측값이 된다.
### Ridge Regression 맛보기
학습이 과대적합 되는 것을 방지하기위해 패널티를 부여한다. (L2 Regularazation)

용어를 몰라도 문서를 보고 다른 알고리즘을 사용하는 방법을 익혀보자.

- 링크 : linear_model의 모델들이 나온다.

https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/5665bd7b-6452-407e-b785-f7841cd1a047)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c0d86cd9-5aa7-4013-8b27-642b60918c35)

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/65beea58-112b-4c20-aa22-b5d6bbced63e)

```python
from sklearn.linear_model import Ridge
// Ridge 모델을 가져온다

model = Ridge()
// 모델 정의

model.fit(x_train, y_train)
// 모델 훈련

y_pred = model.predict(x_val)
// 정답값 예측

mean_absolute_error(y_val, y_pred)
// 예측한 값의 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/d8780000-77e7-4a8e-9b72-a1c6eddfc3f0)

### 당뇨병 데이터셋을 이용한 실습
- 링크 : 당뇨병에 대한 데이터셋이 나온다.

https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c2b16cdd-ab7a-4740-a1ef-a9dd3ad1d112)

#### 데이터셋 로드
- age: 나이
- sex: 성별
- bmi: BMI 체질량지수
- bp: 평균 혈압
- s1 tc: 총 혈청 콜레스테롤
- s2 ldl: 저밀도 지방단백질
- s3 hdl: 고밀도 지방단백질
- s4 tch: 총 콜레스테롤 / HDL
- s5 ltg: 혈청 트리글리세리드 수치의 로그
- s6 glu: 혈당 수치
- target: 1년 후 당뇨병 진행도

```python
from sklearn.datasets import load_diabetes
import pandas as pd
// load_boston 데이터셋 패키지를 가져온다.
// 판다스 라이브러리를 가져온다.

data = load_diabetes()
// load_diabetes를 data안에 데이터 셋 시켜준다.

df = pd.DataFrame(data['data'], columns=data['feature_names'])
// 데이터 프레임에 데이터 값은 load_boston을 데이터셋한 data를 넣어주고, 컬럼은 data의 feature_names로 넣어준다.

df['target'] = data['target']
// 데이터 프레임에 타겟 데이터는 data의 타겟 데이터로 넣어준다.

df.head()
// 상위 데이터 5개를 뽑는다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/379e1ede-1ec4-4182-80cc-6da83303beb8)

#### 데이터 시각화
- 이미 표준화가 되어 있는 데이터셋

```python
sns.boxplot(y=df['age'])
// y 축이 age인 박스 차트를 그린다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/65ac2b9c-3719-4707-9d40-27e07c4fac27)

```python
sns.displot(x=df['target'])
// x축이 타겟인 분포 차트를 그린다.

plt.show()
// 차트를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/c1c60702-d69c-4a27-bb97-ea6d8724a90a)

```python
plt.figure(figsize=(10, 10))
// 가로10 세로10인 차트를 생성한다.

corr = df.corr()
// 누락값을 제외하고 전체 데이터에 상관도를 계산함
sns.heatmap(corr, annot=True, square=True, cmap='PiYG', vmin=-1, vmax=1)
// annot : 수치표시
// square : 틀을 정사각형으로 지정 (false 지정시 직사각형)
// camp : 색상변경
// vmin, vmax : 최대 최소 기준점 변경

plt.show()
차트를 보여준다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/003922a6-c7c4-4e11-8c31-d234746392a9)

#### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=2021)
// train_test_split를 사용하여 x에는 정규화된 데이터를 넣어주고 y에는 타겟을 넣어주고 val은 20%, train은 80%로 랜덤으로 나눈다.

print(x_train.shape, y_train.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.

print(x_val.shape, y_val.shape)
// 몇개씩 나뉘었는지 shape를 사용하여 알아본다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/18921f4a-fb6a-4598-bee2-a638774f49fd)

#### 모델 정의, 학습, 검증
```python
from sklearn.linear_model import SGDRegressor
// SGDRegressor 모델을 가져온다.

model = SGDRegressor()
// 모델정의

model.fit(x_train, y_train)
// 모델 훈련

y_pred = model.predict(x_val)
// 정답값 예측

mean_absolute_error(y_val, y_pred)
// 예측한 값의 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/1d26efe7-48c3-4120-9d08-f3e5e4413fa3)

```python
model = LinearRegression()
// 모델 정의

model.fit(x_train, y_train)
// 모델 훈련

y_pred = model.predict(x_val)
// 정답값 예측

mean_absolute_error(y_val, y_pred)
// 예측한 값의 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/6eeb6162-f645-40dd-aa5a-d5377ce2205a)

#### 검증 결과 시각화
```python
plt.figure(figsize=(8, 6))
// 가로 8 세로 6 차트를 그린다.

sns.scatterplot(x=x_val['bmi'], y=y_val, color='b')
// 정답값을 파란색으로 나타내는 산점도 차트를 그린다.

sns.scatterplot(x=x_val['bmi'], y=y_pred, color='r')
// 예측값을 빨간색으로 나타내는 산점도 차트를 그린다.

plt.show()
// 차트를 나타낸다.
```

![image](https://github.com/hsy0511/bbang-hyung/assets/104752580/b4870b65-7c81-4bf7-9e05-8632dbade10c)
