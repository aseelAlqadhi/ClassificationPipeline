import pandas as pd
import streamlit as st
import joblib

st.set_page_config(layout='wide', page_title='Loan Default Prediction')

st.title('Loan Default Prediction')

# ── Cache the data & model so they load only once ──────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('loan_default.csv')

@st.cache_resource
def load_model():
    return joblib.load('catboost_loan.pkl')

df       = load_data()
ml_model = load_model()

st.dataframe(df.head())

# ── Sidebar inputs ──────────────────────────────────────────────────────────
age             = st.sidebar.slider('Age', min_value=22, max_value=70, step=1)
credit_score    = st.sidebar.slider('Credit Score', min_value=300, max_value=850, step=1)
loan_term       = st.sidebar.selectbox('Loan Term (months)', [12, 24, 36, 48, 60])
education       = st.sidebar.selectbox('Education', ["High School", "Bachelor's", "Master's", "PhD"])
employment_type = st.sidebar.selectbox('Employment Type', sorted(df['employment_type'].dropna().unique()))
marital_status  = st.sidebar.selectbox('Marital Status',  sorted(df['marital_status'].dropna().unique()))

# ── Main inputs ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    income           = st.text_input('Annual Income', '50000')
    loan_amount      = st.text_input('Loan Amount', '10000')

with col2:
    months_employed  = st.text_input('Months Employed', '60')
    num_credit_lines = st.text_input('Number of Credit Lines', '5')

with col3:
    interest_rate    = st.text_input('Interest Rate (%)', '10.0')
    dti_ratio        = st.text_input('Debt-to-Income Ratio', '0.35')

loan_purpose   = st.selectbox('Loan Purpose',  sorted(df['loan_purpose'].dropna().unique()))
has_mortgage   = st.selectbox('Has Mortgage',   ['Yes', 'No'])
has_dependents = st.selectbox('Has Dependents', ['Yes', 'No'])
has_cosigner   = st.selectbox('Has Co-Signer',  ['Yes', 'No'])

# ── Prediction ───────────────────────────────────────────────────────────────
if st.button('Predict Default Risk', type='primary'):
    try:
        new_data = pd.DataFrame({
            'age':              [int(age)],
            'income':           [float(income)],
            'loan_amount':      [float(loan_amount)],
            'credit_score':     [float(credit_score)],
            'months_employed':  [float(months_employed)],
            'num_credit_lines': [float(num_credit_lines)],
            'interest_rate':    [float(interest_rate)],
            'loan_term':        [float(loan_term)],
            'dti_ratio':        [float(dti_ratio)],
            'education':        [education],
            'employment_type':  [employment_type],
            'marital_status':   [marital_status],
            'loan_purpose':     [loan_purpose],
            'has_mortgage':     [has_mortgage],
            'has_dependents':   [has_dependents],
            'has_cosigner':     [has_cosigner],
        })

        prediction = ml_model.predict(new_data)[0]

        if prediction == 1:
            st.error('⚠️ High Risk: This loan is likely to DEFAULT')
        else:
            st.success('✅ Low Risk: This loan is likely to be REPAID')

    except ValueError as e:
        st.warning(f'Please make sure all numeric fields contain valid numbers.\nDetails: {e}')
# import pandas as pd
# import streamlit as st
# import joblib
# from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
# from sklearn.impute import SimpleImputer
# from catboost import CatBoostClassifier

# st.set_page_config(layout='wide', page_title='Loan Default Prediction')

# html_title = '''<h1 style="color:white;text-align:center;"> Loan Default Prediction </h1>'''
# st.markdown(html_title, unsafe_allow_html=True)

# df = pd.read_csv('loan_default.csv')
# st.dataframe(df.head())

# age              = st.sidebar.slider('Age', min_value=22, max_value=70, step=1)
# income           = st.text_input('Annual Income', '')
# loan_amount      = st.text_input('Loan Amount', '')
# credit_score     = st.sidebar.slider('Credit Score', min_value=300, max_value=850, step=1)
# months_employed  = st.text_input('Months Employed', '')
# num_credit_lines = st.text_input('Number of Credit Lines', '')
# interest_rate    = st.text_input('Interest Rate (%)', '')
# loan_term        = st.sidebar.selectbox('Loan Term (months)', [12, 24, 36, 48, 60])
# dti_ratio        = st.text_input('Debt-to-Income Ratio', '')
# education        = st.sidebar.selectbox('Education', ["High School", "Bachelor's", "Master's", "PhD"])
# employment_type  = st.sidebar.selectbox('Employment Type', df['employment_type'].unique())
# marital_status   = st.sidebar.selectbox('Marital Status', df['marital_status'].unique())
# loan_purpose     = st.selectbox('Loan Purpose', df['loan_purpose'].unique())
# has_mortgage     = st.selectbox('Has Mortgage', ['Yes', 'No'])
# has_dependents   = st.selectbox('Has Dependents', ['Yes', 'No'])
# has_cosigner     = st.selectbox('Has Co-Signer', ['Yes', 'No'])

# ml_model = joblib.load('catboost_loan.pkl')

# if st.button('Predict Default Risk'):

#     new_data = pd.DataFrame(columns=df.drop('default', axis=1).columns,
#                             data=[[age, income, loan_amount, credit_score,
#                                    months_employed, num_credit_lines, interest_rate,
#                                    loan_term, dti_ratio, education, employment_type,
#                                    marital_status, loan_purpose, has_mortgage,
#                                    has_dependents, has_cosigner]])

#     prediction = ml_model.predict(new_data)[0]

#     if prediction == 1:
#         st.error('High Risk: This loan is likely to DEFAULT')
#     else:
#         st.success('Low Risk: This loan is likely to be REPAID')





