import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Step 1: Generate synthetic portfolio data
def create_portfolio_data(periods=100, seed=123):
    np.random.seed(seed)
    assets = ['Stocks', 'Bonds', 'RealEstate', 'Commodities']
    XX = {
        'Stocks': np.random.normal(0.05, 1.5, periods),
        'Bonds': np.random.normal(0.03, 1.2, periods),
        'RealEstate': np.random.normal(0.04, 1.8, periods),
        'Commodities': np.random.normal(0.06, 1.3, periods)
    }
    XX['Bonds'] = 0.7 * XX['Bonds'] + 0.3 * XX['Stocks']
    df = pd.DataFrame(XX)
    return df

# Step 2: Calculate correlation matrix
def compute_corr_matrix(df):
    ma = df.corr()
    return ma

# Step 3: Perform clustering analysis
def analyze_clusters(ma):
    clust = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
    dist = 1 - ma
    lbls = clust.fit_predict(dist)
    return lbls

# Step 4: Visualize correlation heatmap
def plot_corr_heatmap(ma, lbls):
    plt.figure(figsize=(8, 6))
    sns.heatmap(ma, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                xticklabels=ma.columns, yticklabels=ma.columns,
                linewidths=0.5, linecolor='gray')
    plt.title('Portfolio Correlation Heatmap', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    clust_info = "\nClusters: " + ", ".join([f"Group {i}" for i in np.unique(lbls)])
    plt.suptitle(clust_info, fontsize=10, y=0.05)
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Step 1: Create data
    df = create_portfolio_data()
    print("Portfolio Data (First 5 Rows):")
    print(df.head())
    
    # Step 2: Compute correlation
    ma = compute_corr_matrix(df)
    
    # Step 3: Analyze clusters
    lbls = analyze_clusters(ma)
    
    # Step 4: Plot heatmap
    plot_corr_heatmap(ma, lbls)