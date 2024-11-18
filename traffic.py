# App to predict the volume of traffic using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from mapie.regression import MapieRegressor
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

#used chatGPT to look up the styling
st.markdown("<h3 style='text-align: center;'>Utilize our advanced Machine Learning application to predict traffic volume.</h3>", unsafe_allow_html=True)
st.image('traffic_image.gif', use_column_width=True)

with open('bst_traffic.pickle', 'rb') as model_pickle:
    bst_model = pickle.load(model_pickle)

st.sidebar.image('traffic_sidebar.jpg', use_column_width=True, caption='Traffic Volume Predictor')

st.sidebar.subheader("Input Features")
st.sidebar.write("You can either upload your data file or manually enter input features.")

df = pd.read_csv('Traffic_Volume.csv').drop(columns=['traffic_volume'])
df['date_time'] = pd.to_datetime(df['date_time'])
df['holiday'] =df['holiday'].fillna('None')
df['month'] = df['date_time'].dt.month_name()
df['weekday'] = df['date_time'].dt.day_name()
df['hour'] = df['date_time'].dt.hour
df = df.drop(columns = ['date_time'])

with st.sidebar.expander("Option 1: Upload CSV File"):
    st.write("Upload a CSV file containing traffic details.")
    file_upload = st.file_uploader("Choose a CSV file", type=['csv'])
    st.subheader("Sample Data Format for Upload")
    st.write(df.head())
    st.warning("⚠️Ensure your uploaded file has the same column names and data types as shown above.")

with st.sidebar.expander("Option 2: Fill out Form"):
    st.write("Enter the traffic details manually using the form below.")
    with st.form("traffic_details_form"):
        holiday = st.selectbox('Choose whether today is a designated holiday or not', options=df['holiday'].unique())
        temp = st.number_input('Average temperature in Kelvin', min_value=float(df['temp'].min()), max_value=float(df['temp'].max()))
        rain = st.number_input('Amount in mm of rain that occured in the hour', min_value=float(df['rain_1h'].min()), max_value=float(df['rain_1h'].max()))
        snow = st.number_input('Amount in mm of snow that occured in the hour', min_value=float(df['snow_1h'].min()), max_value=float(df['snow_1h'].max()))
        clouds = st.number_input('Percentage of cloud cover', min_value=int(df['clouds_all'].min()), max_value=int(df['clouds_all'].max()), step=1)
        weather = st.selectbox('Choose the current weather', options=df['weather_main'].unique())
        month = st.selectbox('Choose month', options=df['month'].unique())
        weekday = st.selectbox('Choose day of the week', options=df['weekday'].unique())
        hour = st.selectbox('Choose hour', options=df['hour'].unique())
        submit_button = st.form_submit_button("Submit Form Data")

if file_upload is None and submit_button:
    st.success("Form completed successfully.")
    alpha = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.5, step=0.01)
    encode_dff = df.copy()
    encode_dff.loc[len(encode_dff)] = [holiday, temp, rain, snow, clouds, weather, month, weekday, hour]
    encode_dff = pd.get_dummies(encode_dff, columns=['holiday','weather_main','month','weekday','hour'])
    user_encoded_dff = encode_dff.tail(1)
    
    st.subheader("Predicting Traffic Volume...")
    st.caption("Predicted Traffic Volume")
    y_test_pred, y_test_pis = bst_model.predict(user_encoded_dff, alpha=alpha)         
    st.subheader(f"{int(y_test_pred[0])}")
    st.write(f"Prediction Interval ({int((1-alpha)*100)}%): [{int(y_test_pis[0][0][0])}, {int(y_test_pis[0][1][0])}]")

elif file_upload is not None:
    st.success("CSV file uploaded successfully.")
    alpha = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.5, step=0.01)
    user_df = pd.read_csv(file_upload)
    user_df = user_df[df.columns] 
    combined_df = pd.concat([df, user_df], axis=0)
    original_rows = df.shape[0]
    combined_df['holiday'] = combined_df['holiday'].fillna("None")

    combined_df_encoded = pd.get_dummies(combined_df, columns=['holiday','weather_main','month','weekday','hour'])
    user_df_encoded = combined_df_encoded[original_rows:]

    user_pred, user_pis = bst_model.predict(user_df_encoded, alpha=alpha)

    user_df["Predicted Volume"] = user_pred.round(0)
    user_df["Lower Limit"] = user_pis[:, 0].round(1)
    user_df["Upper Limit"] = user_pis[:, 1].round(1)

    st.subheader(f"Prediction Results with {int((1-alpha)*100)}% Prediction Interval")
    st.dataframe(user_df)
else:
    st.info("Please choose a data input method to proceed")
    alpha = st.slider('Select alpha value for prediction intervals', min_value=0.01, max_value=0.5, step=0.01)
st.subheader("Model Performance and Inference")

tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")

