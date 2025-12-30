# API Reference

Complete documentation for the Crop Yield Prediction REST API.

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required (prototype).

---

## Endpoints

### Health Check

Check if the API is running and model is loaded.

**GET** `/health`

#### Response

```json
{
  "status": "healthy",
  "message": "API is healthy and model is loaded"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | API is healthy |
| 503 | Model not loaded |

---

### Model Information

Get information about the loaded model.

**GET** `/model/info`

#### Response

```json
{
  "model_version": "Random Forest",
  "supported_crops": ["Wheat", "Maize", "Rice, paddy", ...],
  "supported_countries": ["India", "United States", ...],
  "features": ["rainfall_mm", "pesticides_tonnes", "avg_temp", "crop", "country"]
}
```

---

### Yield Prediction

Predict crop yield for specific conditions.

**POST** `/predict`

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| crop | string | Yes | Crop name (e.g., "Wheat") |
| country | string | Yes | Country name (e.g., "India") |
| rainfall_mm | float | Yes | Average annual rainfall (0-10000 mm) |
| pesticides_tonnes | float | Yes | Pesticides usage (≥0 tonnes) |
| avg_temp | float | Yes | Average temperature (-50 to 60 °C) |

#### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Wheat",
    "country": "India",
    "rainfall_mm": 1000,
    "pesticides_tonnes": 5000,
    "avg_temp": 20
  }'
```

#### Response

```json
{
  "crop": "Wheat",
  "predicted_yield": 26774.26,
  "yield_unit": "hg/ha",
  "model_version": "Random Forest"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Prediction successful |
| 400 | Invalid input (validation error) |
| 503 | Model not loaded |

---

### Crop Recommendation

Get ranked crop recommendations based on conditions.

**POST** `/recommend`

#### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| country | string | Yes | Country name |
| rainfall_mm | float | Yes | Average annual rainfall (0-10000 mm) |
| pesticides_tonnes | float | Yes | Pesticides usage (≥0 tonnes) |
| avg_temp | float | Yes | Average temperature (-50 to 60 °C) |
| top_n | int | No | Number of recommendations (1-20) |

#### Example Request

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "country": "India",
    "rainfall_mm": 1000,
    "pesticides_tonnes": 5000,
    "avg_temp": 20,
    "top_n": 5
  }'
```

#### Response

```json
{
  "recommendations": [
    {
      "rank": 1,
      "crop": "Cassava",
      "predicted_yield": 325060.0,
      "yield_unit": "hg/ha"
    },
    {
      "rank": 2,
      "crop": "Potatoes",
      "predicted_yield": 138580.0,
      "yield_unit": "hg/ha"
    }
  ],
  "context": {
    "country": "India",
    "rainfall_mm": 1000,
    "pesticides_tonnes": 5000,
    "avg_temp": 20
  },
  "model_version": "Random Forest"
}
```

---

## Error Responses

All errors return JSON with a `detail` field:

```json
{
  "detail": "Error message here"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Crop 'X' is not supported" | Invalid crop name | Use a supported crop |
| "rainfall_mm must be non-negative" | Negative value | Use value ≥ 0 |
| "Model not loaded" | Server startup issue | Restart API |

---

## Supported Crops

- Cassava
- Maize
- Plantains and others
- Potatoes
- Rice, paddy
- Sorghum
- Soybeans
- Sweet potatoes
- Wheat
- Yams

---

## Interactive Documentation

When the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
