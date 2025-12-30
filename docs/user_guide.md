# User Guide

How to use the Crop Yield Prediction & Recommendation application.

## Accessing the Application

1. Start the API: `cd api && uvicorn main:app --port 8000`
2. Start Streamlit: `cd app && streamlit run app.py`
3. Open browser: `http://localhost:8501`

---

## Application Modes

### 🔮 Prediction Mode

Predict yield for a specific crop.

#### Steps:

1. Select **🔮 Prediction** from the sidebar
2. Choose your **crop** from the dropdown
3. Select your **country**
4. Set environmental parameters:
   - **Rainfall**: Annual rainfall in mm (slider)
   - **Pesticides**: Usage in tonnes (number input)
   - **Temperature**: Average temp in °C (slider)
5. Click **🔮 Predict Yield**

#### Understanding Results:

- **Predicted Yield**: Shown in hg/ha
- **Conversion**: Also displayed in kg/ha and tonnes/ha

---

### 📊 Recommendation Mode

Get ranked crop recommendations.

#### Steps:

1. Select **📊 Recommendation** from the sidebar
2. Select your **country**
3. Set environmental parameters
4. Choose number of recommendations (top_n)
5. Click **📊 Get Recommendations**

#### Understanding Results:

- **Bar Chart**: Visual comparison of yields
- **Ranking Table**: All crops with predicted yields
- **Top Pick**: Highlighted best recommendation

---

## Input Parameters

### Rainfall (mm/year)

| Range | Category |
|-------|----------|
| 0-500 | Arid |
| 500-1000 | Semi-arid |
| 1000-2000 | Moderate |
| 2000+ | Wet |

### Temperature (°C)

| Range | Category |
|-------|----------|
| <10 | Cold |
| 10-20 | Temperate |
| 20-30 | Warm |
| >30 | Hot |

### Pesticides (tonnes)

National-level pesticide usage. Higher values indicate more intensive agriculture.

---

## Example Scenarios

### Scenario 1: Wheat in India

**Inputs:**
- Crop: Wheat
- Country: India
- Rainfall: 800 mm
- Pesticides: 10,000 tonnes
- Temperature: 25°C

**Expected**: ~25,000-30,000 hg/ha

### Scenario 2: Best Crop for Tropical Climate

**Inputs:**
- Country: Brazil
- Rainfall: 2000 mm
- Pesticides: 15,000 tonnes
- Temperature: 28°C

**Expected top crops**: Cassava, Sweet potatoes, Plantains

---

## Tips

### For Accurate Predictions

1. Use realistic parameter ranges
2. Match country to your actual region
3. Consider local climate averages

### Interpreting Yields

| Yield (hg/ha) | Yield (tonnes/ha) | Interpretation |
|---------------|-------------------|----------------|
| <20,000 | <2 | Low yield |
| 20,000-50,000 | 2-5 | Average |
| 50,000-100,000 | 5-10 | Good |
| >100,000 | >10 | High yield |

---

## Troubleshooting

### "API Not Available"

- Ensure API is running on port 8000
- Check the API URL in the sidebar
- Restart the API server

### Unexpected Results

- Verify input parameters are realistic
- Check if the crop is supported
- Review the model limitations

### Slow Performance

- First prediction may take longer (model loading)
- Subsequent predictions should be fast

---

## Screenshots

### Prediction Screen
*Enter parameters and get yield prediction*

### Recommendation Screen
*View ranked crops with bar chart*

---

## Support

For issues or questions:
1. Check the [API Reference](api_reference.md)
2. Review [Model Documentation](model_documentation.md)
3. Open a GitHub issue
