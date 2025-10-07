# Multimodal Ensemble Architecture for Time Series Forecasting


A cutting-edge multimodal time series forecasting system that combines deep learning, machine learning, and statistical models to predict stock trends and optimize trading strategies. This project was developed as part of the UCI UROP (Undergraduate Research Opportunities Program) Fellowship.

## Key Features

- **Multimodal Architecture**: Integrates CNN + Bi-LSTM, Transformer, XGBoost, and ARIMA models
- **Advanced Feature Engineering**: 20+ technical indicators with FinBERT-based sentiment embeddings
- **Model Interpretability**: SHAP analysis, saliency maps, and temporal attention heatmaps
- **Meta-Ensemble Stacker**: MLP-based stacking for improved predictive accuracy
- **Robust Performance**: R² > 0.85, Huber loss < 0.08, Win rate > 60%, Sharpe ratio > 2.1

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | > 0.85 | Coefficient of determination |
| **Huber Loss** | < 0.08 | Robust regression loss |
| **Win Rate** | > 60% | Trading strategy success rate |
| **Sharpe Ratio** | > 2.1 | Risk-adjusted returns |
| **Accuracy Boost** | 10-15% | Improvement over best single model |
| **Dimensionality Reduction** | 30% | Feature space optimization |

## Architecture Overview

### Model Components

1. **Deep Learning Models**
   - CNN + Bi-LSTM for sequential pattern recognition
   - Transformer architecture for attention-based learning

2. **Machine Learning Models**
   - XGBoost for gradient boosting
   - Statistical ARIMA for time series analysis

3. **Meta-Ensemble Stacker**
   - Multilayer Perceptron (MLP) for model combination
   - Sequential model stacking for enhanced performance

### Feature Pipeline

- **Technical Indicators**: 20+ financial indicators (RSI, MACD, Bollinger Bands, etc.)
- **Sentiment Analysis**: FinBERT-based embeddings for market sentiment
- **Cross-Modal Integration**: Seamless fusion of numerical and textual features
- **Dimensionality Reduction**: 30% reduction through feature importance analysis

## Research Contributions

### Model Interpretability
- **SHAP Analysis**: Quantified feature importance across all modalities
- **Saliency Maps**: Visualized decision mechanisms in deep learning models
- **Temporal Attention Heatmaps**: Elucidated cross-modal reasoning patterns
- **PyTorch Lightning Workflows**: Streamlined model training and analysis

### Ensemble Methodology
- **Meta-Learning Approach**: MLP stacker learns optimal model combinations
- **Variance Reduction**: Significantly reduced prediction variance across folds
- **Cross-Modal Validation**: Ensured robust reasoning across different data modalities


## Technical Stack

- **Deep Learning**: PyTorch, PyTorch Lightning
- **Machine Learning**: XGBoost, scikit-learn
- **Time Series**: ARIMA, pandas, numpy
- **NLP**: FinBERT, transformers
- **Interpretability**: SHAP, captum
- **Visualization**: matplotlib, seaborn, plotly

## Key Results

### Predictive Performance
- Achieved R² > 0.85 on out-of-sample test data
- Maintained Huber loss < 0.08 across different market conditions
- Demonstrated consistent performance across multiple time horizons

### Trading Performance
- Win rate exceeding 60% in backtesting
- Sharpe ratio > 2.1 indicating strong risk-adjusted returns
- Reduced drawdown through ensemble diversification

### Model Improvements
- 10-15% accuracy boost over the strongest single model
- 30% reduction in feature dimensionality without performance loss
- Significant variance reduction across cross-validation folds

## Research Methodology

1. **Data Collection**: Multi-source financial data integration
2. **Feature Engineering**: Technical indicators + sentiment embeddings
3. **Model Development**: Individual model training and validation
4. **Ensemble Construction**: Meta-learning approach for model combination
5. **Interpretability Analysis**: SHAP, attention mechanisms, and saliency maps
6. **Performance Evaluation**: Comprehensive backtesting and risk analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.