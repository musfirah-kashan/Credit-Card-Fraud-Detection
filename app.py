import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Shield", page_icon="🛡️", layout="wide")
 
# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #141824 100%);
    }
    .hero {
        padding: 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        margin-bottom: 2rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1a1f2e;
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 1rem;
    }
    .card {
        background-color: #1a1f2e;
        border: 1px solid #2a3040;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #12151f;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
        color: white;
    }
</style>
""",unsafe_allow_html=True)
model=joblib.load('models/best_model.pkl')
scaler=joblib.load('models/scaler.pkl')
results=pd.read_csv('models/results.csv')

st.sidebar.markdown("## 🛡️ Fraud Shield")
st.sidebar.caption("ML-powered transaction risk detection")
st.sidebar.markdown("---")
page=st.sidebar.radio(
    "Navigate",
    ["🏠 Home","🔍 Single Prediction","📂 Batch Prediction","📊 Model Performance","👤 About Me"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit + Scikit-learn")
 
# ---------------- HOME ----------------
if page=="🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🛡️ Credit Card Fraud Detection</h1>
        <p>Six machine learning models battle-tested with GridSearchCV to catch fraudulent transactions in real time.</p>
    </div>
    """,unsafe_allow_html=True)
 
    best_row=results.sort_values(by='f1',ascending=False).iloc[0]
    col1,col2,col3,col4=st.columns(4)
    col1.metric("🏆 Best Model",best_row['model'])
    col2.metric("F1 Score",f"{best_row['f1']:.3f}")
    col3.metric("ROC-AUC",f"{best_row['roc_auc']:.3f}")
    col4.metric("Models Trained",len(results))
 
    st.markdown("<br>",unsafe_allow_html=True)
 
    colA,colB=st.columns(2)
    with colA:
        st.markdown("""
        <div class="card">
            <h4>🔍 Single Prediction</h4>
            <p>Test one transaction manually using amount and time. Get an instant fraud risk score.</p>
        </div>
        """,unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <h4>📊 Model Performance</h4>
            <p>Compare all 6 trained models side by side with F1 and ROC-AUC scores.</p>
        </div>
        """,unsafe_allow_html=True)
    with colB:
        st.markdown("""
        <div class="card">
            <h4>📂 Batch Prediction</h4>
            <p>Upload a CSV of many transactions and download fraud predictions for all of them at once.</p>
        </div>
        """,unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <h4>🧠 Models Trained</h4>
            <p>{", ".join(results['model'].tolist())}</p>
        </div>
        """,unsafe_allow_html=True)

elif page=="🔍 Single Prediction":
    st.markdown("## 🔍 Check a Single Transaction")
    st.caption("Enter transaction details below to get a fraud risk score.")
    st.markdown("<br>",unsafe_allow_html=True)
 
    col1,col2=st.columns(2)
    with col1:
        amount=st.slider('💵 Transaction Amount ($)',0.0,25000.0,100.0)
    with col2:
        time=st.slider('⏱️ Time (seconds since first transaction)',0.0,172800.0,50000.0)
 
    st.markdown("#### Behavior Pattern Features")
    st.caption("These anonymized features (V1-V28) actually drive fraud detection more than Amount/Time. Push them further from 0 to simulate riskier behavior patterns.")
 
    c1,c2,c3,c4,c5=st.columns(5)
    with c1:
        v14=st.slider('V14',-20.0,10.0,0.0)
    with c2:
        v4=st.slider('V4',-5.0,17.0,0.0)
    with c3:
        v12=st.slider('V12',-19.0,8.0,0.0)
    with c4:
        v10=st.slider('V10',-19.0,24.0,0.0)
    with c5:
        v17=st.slider('V17',-25.0,10.0,0.0)
 
    st.markdown("<br>",unsafe_allow_html=True)
    check=st.button('🚀 Check Transaction',use_container_width=True)
 
    if check:
        amount_scaled=scaler.transform([[amount]])[0][0]
        time_scaled=scaler.transform([[time]])[0][0]
        v_features=[0.0]*28
        v_features[13]=v14   
        v_features[3]=v4    
        v_features[11]=v12   
        v_features[9]=v10    
        v_features[16]=v17   
        input_data=np.array([v_features+[amount_scaled,time_scaled]])
 
        prediction=model.predict(input_data)[0]
        probability=model.predict_proba(input_data)[0][1]
 
        st.markdown("<br>",unsafe_allow_html=True)
        r1,r2=st.columns([2,1])
        with r1:
            if prediction==1:
                st.error(f"🚨 **Fraud Detected** — Risk Score: {probability*100:.2f}%")
            else:
                st.success(f"✅ **Transaction Looks Safe** — Risk Score: {probability*100:.2f}%")
            st.progress(min(int(probability*100),100))
        with r2:
            st.metric("Risk Score",f"{probability*100:.1f}%")
 
# ---------------- BATCH PREDICTION ----------------
elif page=="📂 Batch Prediction":
    st.markdown("## 📂 Check Many Transactions")
    st.caption("Upload a CSV with the same columns as training data (Time, V1-V28, Amount). No 'Class' column needed.")
    st.markdown("<br>",unsafe_allow_html=True)
 
    file=st.file_uploader("Drag and drop a CSV file here",type=['csv'])
 
    if file is not None:
        data=pd.read_csv(file)
        st.markdown("#### Preview")
        st.dataframe(data.head(),use_container_width=True)
 
        run=st.button('⚡ Run Predictions on File',use_container_width=True)
 
        if run:
            data['Amount_scaled']=scaler.transform(data[['Amount']])
            data['Time_scaled']=scaler.transform(data[['Time']])
            data_model=data.drop(['Amount','Time'],axis=1)
 
            predictions=model.predict(data_model)
            probabilities=model.predict_proba(data_model)[:,1]
 
            data['Prediction']=np.where(predictions==1,'Fraud','Not Fraud')
            data['Risk_Score']=(probabilities*100).round(2)
 
            fraud_count=(predictions==1).sum()
 
            st.markdown("<br>",unsafe_allow_html=True)
            m1,m2,m3=st.columns(3)
            m1.metric("Total Transactions",len(data))
            m2.metric("🚨 Fraud Detected",int(fraud_count))
            m3.metric("✅ Safe",len(data)-int(fraud_count))
 
            st.markdown("#### Results")
            st.dataframe(data[['Prediction','Risk_Score']+[c for c in data.columns if c not in ['Prediction','Risk_Score']]],use_container_width=True)
 
            csv_download=data.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV",csv_download,"fraud_predictions.csv","text/csv",use_container_width=True)
 
# ---------------- MODEL PERFORMANCE ----------------
elif page=="📊 Model Performance":
    st.markdown("## 📊 Model Performance Comparison")
    st.caption("How each of the 6 trained models performed on the test set.")
    st.markdown("<br>",unsafe_allow_html=True)
 
    st.dataframe(results.sort_values(by='f1',ascending=False).reset_index(drop=True),use_container_width=True)
 
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### F1 Score by Model")
        fig,ax=plt.subplots()
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        bars=ax.bar(results['model'],results['f1'],color='#6a11cb')
        ax.tick_params(colors='white')
        ax.set_ylabel('F1 Score',color='white')
        for spine in ax.spines.values():
            spine.set_color('#444')
        plt.xticks(rotation=45,color='white')
        st.pyplot(fig)
    with c2:
        st.markdown("#### ROC-AUC by Model")
        fig2,ax2=plt.subplots()
        fig2.patch.set_alpha(0)
        ax2.set_facecolor('none')
        ax2.bar(results['model'],results['roc_auc'],color='#2575fc')
        ax2.tick_params(colors='white')
        ax2.set_ylabel('ROC-AUC',color='white')
        for spine in ax2.spines.values():
            spine.set_color('#444')
        plt.xticks(rotation=45,color='white')
        st.pyplot(fig2)
 
    best_row=results.sort_values(by='f1',ascending=False).iloc[0]
    st.markdown(f"""
    <div class="card">
        <h4>🏆 Best Model: {best_row['model']}</h4>
        <p>F1 Score: {best_row['f1']:.4f} &nbsp;|&nbsp; ROC-AUC: {best_row['roc_auc']:.4f}</p>
    </div>
    """,unsafe_allow_html=True)
 
# ---------------- ABOUT ME ----------------
elif page=="👤 About Me":
    st.markdown("## 👤 About Me")
    st.markdown("<br>",unsafe_allow_html=True)
 
    col1,col2=st.columns([1,3])
    with col1:
        try:
            st.image("images/profile.jpg",width=200)
        except Exception:
            st.warning("Add a photo named 'profile.jpg' to your project folder to show it here.")
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Musfirah Kashan</h3>
            <p>Aspiring Machine Learning Engineer</p>
            <p>I build machine learning projects to practice data preprocessing, model training, and deployment through interactive apps.</p>
        </div>
        """,unsafe_allow_html=True)
 
    c1,c2=st.columns(2)
    with c1:
        st.markdown("""
        <div class="card">
            <h4>🛠️ Skills</h4>
            <p>Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn</p>
        </div>
        """,unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
            <h4>📁 Projects</h4>
            <p>Student Performance Prediction (Regression)<br>Credit Card Fraud Detection (Classification)</p>
        </div>
        """,unsafe_allow_html=True)
 
    st.markdown("""
    <div class="card">
        <h4>🔗 Connect With Me</h4>
        <p>GitHub: <a href="https://github.com/musfirah-kashan" style="color:#2575fc;">https://github.com/musfirah-kashan</a><br>
        LinkedIn: <a href="https://www.linkedin.com/in/musfirah-kashan-487aa626a/" style="color:#2575fc;">https://www.linkedin.com/in/musfirah-kashan-487aa626a/</a><br>
        Email: your-email@example.com</p>
    </div>
    """,unsafe_allow_html=True)
