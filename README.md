a# Demand Forecasting Library

> Production-style time-series forecasting for demand and rate prediction — incorporating seasonality, inventory scarcity signals, and competitor dynamics.

---

## Overview

This library provides a modular, extensible demand forecasting system built on top of Prophet, AtsPy, and scikit-learn. It was designed to mirror real-world forecasting challenges in inventory-based businesses (hotels, subscriptions, e-commerce) where accurate forward-looking predictions directly influence pricing strategy and commercial decisions.

**Key capabilities:**
- Multi-model forecasting with automatic model selection
- Seasonality decomposition (weekly, monthly, annual)
- Inventory scarcity signal integration
- Competitor price signal incorporation
- Confidence interval generation for scenario planning
- Automated retraining pipeline

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core forecasting | Prophet, AtsPy, scikit-learn |
| Data processing | pandas, numpy, PySpark |
| Visualisation | Plotly, matplotlib |
| Pipeline orchestration | Airflow-compatible |
| Data storage | PostgreSQL, Snowflake-compatible |
| Version control | Git, dbt for transformations |

---

## Project Structure

```
demand-forecasting/
├── data/
│   ├── sample_demand_data.csv
│   └── README.md
├── src/
│   ├── forecasters/
│   │   ├── prophet_forecaster.py
│   │   ├── atspy_forecaster.py
│   │   └── ensemble_forecaster.py
│   ├── features/
│   │   ├── seasonality.py
│   │   ├── scarcity_signals.py
│   │   └── competitor_signals.py
│   ├── pipeline/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── utils/
│       ├── data_loader.py
│       └── metrics.py
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_business_interpretation.ipynb
├── tests/
│   └── test_forecasters.py
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/priyankasinhabhu/demand-forecasting.git
cd demand-forecasting

# Install dependencies
pip install -r requirements.txt

# Run a basic forecast
python src/pipeline/train.py --data data/sample_demand_data.csv --horizon 30

# Generate forecast with visualisation
python src/pipeline/predict.py --plot
```

---

## Example Output

```python
from src.forecasters.ensemble_forecaster import EnsembleForecaster

forecaster = EnsembleForecaster(models=['prophet', 'atspy'], horizon=30)
forecaster.fit(df, date_col='ds', target_col='y')
forecast = forecaster.predict()
forecaster.plot(forecast)
```

The ensemble model automatically selects the best-performing model based on cross-validated MAPE and returns confidence intervals for scenario planning.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAPE | Mean Absolute Percentage Error |
| RMSE | Root Mean Squared Error |
| Coverage | Confidence interval coverage at 80% and 95% |

---

## Business Context

This library was inspired by real-world demand forecasting challenges in B2B travel and inventory management, where:
- **ADR (Average Daily Rate)** prediction directly influences pricing guardrails
- **Room Night demand** forecasting supports inventory allocation decisions
- **Seasonality and scarcity** signals are critical for yield optimisation

The same methodology applies to e-commerce inventory, subscription churn forecasting, and any time-sensitive demand planning use case.

---

## Author

**Priyanka Sinha** — Senior Data Scientist & Analytics Leader  
📍 Cologne, Germany | [LinkedIn](https://linkedin.com/in/priyanka-sinha) | [Email](mailto:priyankasinhabhu@gmail.com)

---

## License

MIT License — free to use, adapt, and build upon with attribution.
