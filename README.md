# Asset Correlation Heatmap Generator

- This Python tool calculates and visualizes the correlation matrix of asset returns across a lending portfolio (e.g., stocks, bonds, real estate, commodities).
- It includes clustering to identify diversification benefits or systemic risk clusters within the portfolio.

---

## Files
- `asset_correlation_heatmap.py`: Main script for generating synthetic portfolio data, computing correlations, clustering assets, and plotting a heatmap.
- `output.png`: Plot.
---

## Libraries Used
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `sklearn.cluster`

---

## Features
- **Data Generation**: Creates synthetic portfolio returns for multiple asset classes with realistic correlations (e.g., bonds partially tied to stocks).
- **Correlation Matrix**: Computes pairwise correlations between asset returns using pandas.
- **Clustering**: Applies Agglomerative Clustering to group assets based on correlation distance, highlighting diversification or risk clusters.
- **Visualization**: Generates a heatmap with Seaborn, annotated with correlation values and cluster information.
