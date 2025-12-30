"""
Streamlit application for Crop Yield Prediction and Recommendation.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

# Configuration
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction & Recommendation",
    page_icon="🌾",
    layout="wide"
)

# Title and description
st.title("🌾 Crop Yield Prediction & Recommendation System")
st.markdown("""
This application helps farmers predict expected crop yields and get recommendations 
for the most suitable crops based on environmental conditions.
""")

# Sidebar - Mode selection
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Select Mode",
    ["🔮 Prediction", "📊 Recommendation", "ℹ️ About"]
)


def check_api_health() -> bool:
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info() -> Optional[dict]:
    """Get model information from the API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def make_prediction(crop: str, country: str, rainfall: float, pesticides: float, temp: float) -> Optional[dict]:
    """Make a yield prediction via the API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "crop": crop,
                "country": country,
                "rainfall_mm": rainfall,
                "pesticides_tonnes": pesticides,
                "avg_temp": temp
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the API is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None


def get_recommendations(country: str, rainfall: float, pesticides: float, temp: float, top_n: int = 10) -> Optional[dict]:
    """Get crop recommendations via the API."""
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={
                "country": country,
                "rainfall_mm": rainfall,
                "pesticides_tonnes": pesticides,
                "avg_temp": temp,
                "top_n": top_n
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the API is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return None


# Default values
DEFAULT_CROPS = [
    "Wheat", "Maize", "Rice, paddy", "Potatoes", "Soybeans",
    "Sorghum", "Cassava", "Sweet potatoes", "Plantains and others", "Yams"
]

DEFAULT_COUNTRIES = [
    "India", "United States", "China", "Brazil", "Russia",
    "France", "Germany", "Australia", "Canada", "Argentina"
]


# Check API status
api_status = check_api_health()
if api_status:
    st.sidebar.success("✅ API Connected")
    model_info = get_model_info()
    if model_info:
        CROPS = model_info.get("supported_crops", DEFAULT_CROPS)
        COUNTRIES = model_info.get("supported_countries", DEFAULT_COUNTRIES)
    else:
        CROPS = DEFAULT_CROPS
        COUNTRIES = DEFAULT_COUNTRIES
else:
    st.sidebar.error("❌ API Not Available")
    CROPS = DEFAULT_CROPS
    COUNTRIES = DEFAULT_COUNTRIES


# Main content based on mode
if mode == "🔮 Prediction":
    st.header("🔮 Crop Yield Prediction")
    st.markdown("Predict the expected yield for a specific crop given environmental conditions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        # Crop selection
        selected_crop = st.selectbox(
            "Select Crop",
            options=CROPS,
            help="Choose the crop you want to predict yield for"
        )
        
        # Country selection
        selected_country = st.selectbox(
            "Select Country",
            options=COUNTRIES,
            help="Choose the country/region"
        )
        
        # Environmental parameters
        rainfall = st.slider(
            "Average Rainfall (mm/year)",
            min_value=0,
            max_value=5000,
            value=1000,
            step=50,
            help="Average annual rainfall in millimeters"
        )
        
        pesticides = st.number_input(
            "Pesticides Usage (tonnes)",
            min_value=0.0,
            max_value=500000.0,
            value=5000.0,
            step=100.0,
            help="Total pesticides usage in tonnes"
        )
        
        avg_temp = st.slider(
            "Average Temperature (°C)",
            min_value=-10.0,
            max_value=45.0,
            value=20.0,
            step=0.5,
            help="Average annual temperature in Celsius"
        )
        
        predict_button = st.button("🔮 Predict Yield", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Prediction Result")
        
        if predict_button:
            if not api_status:
                st.warning("API is not available. Please start the API first.")
            else:
                with st.spinner("Making prediction..."):
                    result = make_prediction(
                        crop=selected_crop,
                        country=selected_country,
                        rainfall=rainfall,
                        pesticides=pesticides,
                        temp=avg_temp
                    )
                
                if result:
                    # Display prediction
                    st.success("Prediction Complete!")
                    
                    # Big metric display
                    st.metric(
                        label=f"Predicted Yield for {result['crop']}",
                        value=f"{result['predicted_yield']:,.0f} {result['yield_unit']}"
                    )
                    
                    # Additional info
                    st.info(f"""
                    **Details:**
                    - Crop: {result['crop']}
                    - Unit: {result['yield_unit']} (hectograms per hectare)
                    - Model: {result['model_version']}
                    """)
                    
                    # Convert to more common units
                    yield_kg_ha = result['predicted_yield'] / 10
                    yield_tonnes_ha = yield_kg_ha / 1000
                    
                    st.markdown(f"""
                    **Yield in other units:**
                    - {yield_kg_ha:,.0f} kg/ha
                    - {yield_tonnes_ha:,.2f} tonnes/ha
                    """)
        else:
            st.info("👈 Enter parameters and click 'Predict Yield' to see results")


elif mode == "📊 Recommendation":
    st.header("📊 Crop Recommendation")
    st.markdown("Get recommendations for the best crops to grow based on your environmental conditions.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Country selection
        rec_country = st.selectbox(
            "Select Country",
            options=COUNTRIES,
            key="rec_country",
            help="Choose your country/region"
        )
        
        # Environmental parameters
        rec_rainfall = st.slider(
            "Average Rainfall (mm/year)",
            min_value=0,
            max_value=5000,
            value=1000,
            step=50,
            key="rec_rainfall"
        )
        
        rec_pesticides = st.number_input(
            "Pesticides Usage (tonnes)",
            min_value=0.0,
            max_value=500000.0,
            value=5000.0,
            step=100.0,
            key="rec_pesticides"
        )
        
        rec_temp = st.slider(
            "Average Temperature (°C)",
            min_value=-10.0,
            max_value=45.0,
            value=20.0,
            step=0.5,
            key="rec_temp"
        )
        
        top_n = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=len(CROPS),
            value=min(5, len(CROPS)),
            key="top_n"
        )
        
        recommend_button = st.button("📊 Get Recommendations", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Recommendations")
        
        if recommend_button:
            if not api_status:
                st.warning("API is not available. Please start the API first.")
            else:
                with st.spinner("Getting recommendations..."):
                    result = get_recommendations(
                        country=rec_country,
                        rainfall=rec_rainfall,
                        pesticides=rec_pesticides,
                        temp=rec_temp,
                        top_n=top_n
                    )
                
                if result:
                    st.success("Recommendations Ready!")
                    
                    # Convert to DataFrame
                    recommendations = result['recommendations']
                    df = pd.DataFrame(recommendations)
                    
                    # Bar chart
                    fig = px.bar(
                        df,
                        x='predicted_yield',
                        y='crop',
                        orientation='h',
                        title='Predicted Yields by Crop',
                        labels={
                            'predicted_yield': 'Predicted Yield (hg/ha)',
                            'crop': 'Crop'
                        },
                        color='predicted_yield',
                        color_continuous_scale='Greens'
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.subheader("Ranking Table")
                    display_df = df[['rank', 'crop', 'predicted_yield', 'yield_unit']].copy()
                    display_df.columns = ['Rank', 'Crop', 'Predicted Yield', 'Unit']
                    display_df['Predicted Yield'] = display_df['Predicted Yield'].apply(lambda x: f"{x:,.0f}")
                    
                    st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Top recommendation highlight
                    top_crop = recommendations[0]
                    st.success(f"""
                    🏆 **Top Recommendation: {top_crop['crop']}**
                    
                    Based on your conditions (Rainfall: {rec_rainfall}mm, Temperature: {rec_temp}°C), 
                    **{top_crop['crop']}** is predicted to have the highest yield of 
                    **{top_crop['predicted_yield']:,.0f} {top_crop['yield_unit']}**.
                    """)
        else:
            st.info("👈 Enter parameters and click 'Get Recommendations' to see results")


elif mode == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ## Crop Yield Prediction & Recommendation System
    
    This application is designed to help farmers and agricultural planners make informed decisions 
    about crop selection and yield expectations.
    
    ### Features
    
    - **🔮 Yield Prediction**: Predict the expected yield for a specific crop based on:
        - Country/Region
        - Average annual rainfall
        - Pesticides usage
        - Average temperature
    
    - **📊 Crop Recommendation**: Get a ranked list of crops sorted by predicted yield 
      for your specific environmental conditions.
    
    ### How It Works
    
    1. The system uses a machine learning model trained on historical agricultural data
    2. The model considers environmental factors (rainfall, temperature) and agricultural inputs (pesticides)
    3. Predictions are made for yield in hectograms per hectare (hg/ha)
    
    ### Data Sources
    
    - FAO (Food and Agriculture Organization) crop yield data
    - Climate and weather datasets
    - Pesticide usage statistics
    
    ### Model Information
    """)
    
    if api_status and model_info:
        st.json({
            "Model Version": model_info.get("model_version", "N/A"),
            "Number of Supported Crops": len(model_info.get("supported_crops", [])),
            "Features Used": model_info.get("features", [])
        })
    else:
        st.warning("Connect to the API to see model information")
    
    st.markdown("""
    ### Disclaimer
    
    This tool provides estimates based on historical data and machine learning predictions. 
    Actual yields may vary due to factors not captured in the model, including:
    - Soil quality and composition
    - Specific weather events
    - Pest and disease outbreaks
    - Farming practices and techniques
    - Market conditions
    
    Always consult with local agricultural experts for critical farming decisions.
    
    ---
    
    **Version**: 1.0.0  
    **School Project**: Crop Yield Prediction System
    """)


# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        🌾 Crop Yield Prediction System | Built with Streamlit & FastAPI
    </div>
    """,
    unsafe_allow_html=True
)
