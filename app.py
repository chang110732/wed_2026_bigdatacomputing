import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 웹페이지 넓게 쓰기 설정
st.set_page_config(layout="wide")
st.title("기대수명 예측 및 규제 모델 성능 비교")
st.write("WHO 기대수명 데이터셋을 활용한 회귀 모델 대시보드")

# 2단계에서 저장해둔 데이터랑 모델들 불러오기
metrics_df = pd.read_csv('metrics.csv')
models = {
    'Linear': joblib.load('model_linear.pkl'), 
    'Poly': joblib.load('model_poly.pkl'), 
    'Ridge': joblib.load('model_ridge.pkl')
}

# --- [조건 3] 모델 성능 비교 화면 ---
st.header("1. 모델 성능 비교 (과대적합 관찰)")

st.subheader("성능 평가지표 테이블")
st.dataframe(metrics_df)

st.subheader("Test R² 점수 비교 차트")
fig, ax = plt.subplots(figsize=(7, 3))
sns.barplot(x='Model', y='Test R2', data=metrics_df, ax=ax, palette='Set2')
ax.set_ylim(-1.5, 1.0) # 오버피팅으로 점수 떨어지는 것 대비
st.pyplot(fig)

st.markdown("---")

# 실시간 예측 UI 구성
st.header("2. 실시간 기대수명 예측")

# 사이드바에 슬라이더 배치
st.sidebar.header("변수 조정 슬라이더")
am = st.sidebar.slider("Adult mortality", 1, 750, 150)
bmi = st.sidebar.slider("BMI", 1.0, 80.0, 30.0)
gdp = st.sidebar.slider("GDP", 1, 120000, 5000)

# 드롭다운으로 모델 선택
sel_model = st.selectbox("예측에 사용할 모델을 선택하세요", ["Linear", "Poly", "Ridge"])

# 예측용 데이터프레임 만들기 (2단계에서 쓴 컬럼명과 똑같이 일치시킴)
in_df = pd.DataFrame([{'Adult mortality': am, 'BMI': bmi, 'GDP': gdp}])

# 선택한 모델로 예측값 계산
pred = models[sel_model].predict(in_df)[0]

# 결과 출력
st.subheader(f"{sel_model} 모델의 예상 기대수명: {pred:.2f} 세")
