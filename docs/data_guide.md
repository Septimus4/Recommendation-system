# Data Guide

Documentation for datasets used in the Crop Yield Prediction system.

## Dataset Overview

### Source Files

| File | Description | Rows |
|------|-------------|------|
| yield.csv | FAO crop yield data | ~57,000 |
| yield_df.csv | Pre-merged yield with context | ~28,000 |
| pesticides.csv | Pesticide usage by country | ~4,300 |
| rainfall.csv | Rainfall by country/year | ~6,700 |
| temp.csv | Temperature by country/year | ~71,000 |

### Consolidated Dataset

**Location**: `data/processed/consolidated.csv`

| Column | Type | Description |
|--------|------|-------------|
| country | string | Country name |
| crop | string | Crop type |
| year | int | Year (1990-2013) |
| yield | float | Crop yield in hg/ha |
| rainfall_mm | float | Annual rainfall (mm) |
| pesticides_tonnes | float | Pesticide usage (tonnes) |
| avg_temp | float | Average temperature (°C) |

**Statistics**:
- Total rows: 13,130
- Countries: 101
- Crops: 10
- Year range: 1990-2013

---

## Data Processing

### Column Mapping

| Original | Standardized |
|----------|-------------|
| Area | country |
| Item | crop |
| Year | year |
| hg/ha_yield | yield |
| average_rain_fall_mm_per_year | rainfall_mm |
| pesticides_tonnes | pesticides_tonnes |
| avg_temp | avg_temp |

### Cleaning Steps

1. **Strip whitespace** from text columns
2. **Convert year** to integer
3. **Impute missing values** with column median
4. **Remove duplicates** based on (country, crop, year)

### Missing Value Handling

| Column | Strategy |
|--------|----------|
| rainfall_mm | Median imputation |
| pesticides_tonnes | Median imputation |
| avg_temp | Median imputation |
| yield | Drop row (target) |

---

## Crops Included

| Crop | Records | Avg Yield (hg/ha) |
|------|---------|-------------------|
| Cassava | 1,327 | ~100,000 |
| Maize | 2,157 | ~40,000 |
| Plantains | 790 | ~60,000 |
| Potatoes | 1,756 | ~150,000 |
| Rice, paddy | 1,719 | ~35,000 |
| Sorghum | 1,494 | ~12,000 |
| Soybeans | 998 | ~18,000 |
| Sweet potatoes | 1,163 | ~90,000 |
| Wheat | 1,542 | ~25,000 |
| Yams | 184 | ~100,000 |

---

## Feature Descriptions

### Yield (Target)

- **Unit**: hg/ha (hectograms per hectare)
- **Conversion**: 1 hg/ha = 0.1 kg/ha = 0.0001 tonnes/ha
- **Range**: 0 - 900,000+ hg/ha

### Rainfall

- **Unit**: mm/year
- **Description**: Average annual precipitation
- **Typical range**: 0 - 5,000 mm

### Pesticides

- **Unit**: tonnes of active ingredients
- **Description**: Total national pesticide usage
- **Note**: Country-level aggregate, not per-hectare

### Temperature

- **Unit**: °C (Celsius)
- **Description**: Average annual temperature
- **Typical range**: -10 to 35°C

---

## Data Quality Notes

### Limitations

1. **Temporal coverage**: Data ends at 2013
2. **Spatial granularity**: Country-level only
3. **Pesticide proxy**: National totals, not crop-specific
4. **Missing soil data**: Soil quality not included

### Known Issues

- Some country names vary between datasets
- Pesticide values are proxies for input intensity
- Temperature/rainfall may not capture seasonal variation

---

## Regenerating the Dataset

To regenerate `consolidated.csv`:

```python
# Run the EDA notebook
jupyter notebook notebooks/01_eda_and_fusion.ipynb
```

Or via Python:

```python
import pandas as pd
from pathlib import Path

# Load raw data
df = pd.read_csv('data/raw/yield_df.csv', index_col=0)

# Rename columns
df = df.rename(columns={
    'Area': 'country',
    'Item': 'crop',
    'Year': 'year',
    'hg/ha_yield': 'yield',
    'average_rain_fall_mm_per_year': 'rainfall_mm'
})

# Clean
df['country'] = df['country'].str.strip()
df['crop'] = df['crop'].str.strip()
df['year'] = df['year'].astype(int)

# Impute missing
for col in ['rainfall_mm', 'pesticides_tonnes', 'avg_temp']:
    df[col] = df[col].fillna(df[col].median())

# Remove duplicates
df = df.drop_duplicates(subset=['country', 'crop', 'year'])

# Save
df.to_csv('data/processed/consolidated.csv', index=False)
```
