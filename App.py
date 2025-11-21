# ======== Import Libraries ========
import streamlit as st
import pandas as pd
import joblib

# ======== Page Configuration ========
st.set_page_config(layout='wide', page_title='🏠 House Price Prediction')


# # # عنوان وصورة فوق في النص
# # top_col1, top_col2, top_col3 = st.columns([2, 8, 2])
# # with top_col2:
# #     st.image("housingcosts-1024x626-1.jpg", use_container_width=True)

# html_title = """<h1 style="color:white;text-align:center;">🏡 Melbourne House Price Prediction</h1>"""
# st.markdown(html_title, unsafe_allow_html=True)

# ======== Title + Description ========
html_title = """
<h1 style="
    color: white;
    text-align: center;
    margin-bottom: 0.2rem;
">
    🏡 Melbourne House Price Prediction
</h1>
"""
st.markdown(html_title, unsafe_allow_html=True)

html_desc = """
<div style="text-align: center;">
  <p style="
      font-size: 18px;
      color: #f5f5f5;
      max-width: 900px;
      margin: 0 auto 1.5rem auto;
  ">
    Welcome to the <b>Melbourne House Price Predictor App</b> — a simple and interactive tool
    that helps you estimate the price of a house in Melbourne based on real historical market data.
  </p>
</div>
"""
st.markdown(html_desc, unsafe_allow_html=True)






# ======== Global Style (Background + Centered Inputs) ========
st.markdown(
    """
    <style>
    /* خلفية الصفحة */
    [data-testid="stAppViewContainer"] {
        background-image:
            linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.55)),
            url("https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1400&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.8);
    }

    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
        text-shadow: 1px 1px 3px black;
    }

    /* تكبير الزر وتنسيقه */
    div.stButton > button:first-child {
        background-color: #4CAF50; 
        color: white;              
        border-radius: 12px;       
        height: 3em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;  
        transform: scale(1.03);
    }

    /* تصغير وعمل سنتر لكل الـ inputs */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stSlider"] {
        width: 100% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======== Import Data And Model ========
df = pd.read_csv('cleaned_data.csv')   # suburb هيبقى عمود عادي
model = joblib.load('model.pkl')

# ======== Inputs In Center Column ========
left, center, right = st.columns([1, 2, 1])

with center:
    suburb = st.selectbox("📍 Select suburb :", df['suburb'].unique())

    rooms = st.number_input(
        "🛏️ Enter number of rooms :",
        min_value=int(df["rooms"].min()),
        max_value=int(df["rooms"].max()),
        value=int(df["rooms"].mean())
    )

    type_ = st.selectbox("🏠 Select property type :", df["type"].unique())

    method = st.selectbox("💰 Select sale method :", df["method"].unique())

    sellerG = st.selectbox("🤝 Select seller (agent) :", df["sellerg"].unique())

    distance = st.number_input(
        "📏 Enter distance from CBD (km) :",
        min_value=0.0,
        max_value=50.0,
        step=0.1
    )

    bedroom2 = st.slider(
        "🛌 Enter number of bedrooms :",
        min_value=int(df["bedroom2"].min()),
        max_value=int(df["bedroom2"].max()),
        step=1
    )

    bathroom = st.slider(
        "🛁 Enter number of bathrooms :",
        min_value=int(df["bathroom"].min()),
        max_value=int(df["bathroom"].max()),
        step=1
    )

    car = st.slider(
        "🚗 Enter number of car spots :",
        min_value=int(df["car"].min()),
        max_value=int(df["car"].max()),
        step=1
    )

    landsize = st.number_input(
        "🌿 Enter land size (m²) :",
        min_value=float(df["landsize"].min()),
        max_value=float(df["landsize"].max()),
        step=10.0
    )

    yearbuilt = st.slider(
        "🏗️ Enter the year the property was built :",
        min_value=int(df["yearbuilt"].min()),
        max_value=int(df["yearbuilt"].max()),
        step=1
    )

    councilarea = st.selectbox(
        "🏛️ Select the local council area :",
        df["councilarea"].unique()
    )

    regionname = st.selectbox(
        "🌆 Select the property region :",
        sorted(df["regionname"].unique())
    )

    year = st.selectbox("📆 Select sale year :", sorted(df["year"].unique()))

    month = st.selectbox("🗓️ Select sale month :", sorted(df["month"].unique()))

    day = st.selectbox("📅 Select sale day :", sorted(df["day"].unique()))

    season = st.selectbox("☀️ Select season :", sorted(df["season"].unique()))

# ======== Generate one row DataFrame ========
input_columns = [
    'suburb', 'rooms', 'type', 'method', 'sellerg', 'distance',
    'bedroom2', 'bathroom', 'car', 'landsize', 'yearbuilt',
    'councilarea', 'regionname', 'year', 'month', 'day', 'season'
]

new_data = pd.DataFrame(
    [[suburb, rooms, type_, method, sellerG, distance, bedroom2,
      bathroom, car, landsize, yearbuilt, councilarea, regionname,
      year, month, day, season]],
    columns=input_columns
)

# ======== Centered Button Style ========
center_button = """
<style>
div.stButton > button {
    display: block;
    margin-left: auto;
    margin-right: auto;
    background-color: #4CAF50;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 200px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #45a049;
    transform: scale(1.03);
}
</style>
"""

st.markdown(center_button, unsafe_allow_html=True)

# ======== Actual Button ========
if st.button("💸 Predict Price"):
    result = model.predict(new_data)
    st.success(f"🏠 Estimated House Price: ${result[0]:,.2f}")
