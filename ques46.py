# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create the dataset
data = {
    'CustomerID': [101,102,103,104,105,106,107,108,109,110],
    'Annual_Income': [30000, 45000, 50000, 70000, 35000, 85000, 60000, 40000, 75000, 30000],
    'Spending_Score': [40, 65, 20, 90, 50, 70, 80, 55, 95, 30],
    'Age': [25, 30, 22, 40, 28, 45, 35, 33, 50, 20],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'Dallas', 'Seattle', 'Denver', 'San Diego', 'Boston']
}

df = pd.DataFrame(data)

# -----------------------------
# 1. EDA: Scatter plot
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual_Income', y='Spending_Score', data=df, s=100, color='blue')
plt.title('Customer Spending Score vs Annual Income')
plt.xlabel('Annual Income (USD)')
plt.ylabel('Spending Score (1-100)')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Prepare data for clustering
# -----------------------------
X = df[['Annual_Income', 'Spending_Score']]

# -----------------------------
# 3. Elbow Method to find K
# -----------------------------
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of Clusters K')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Apply KMeans with optimal K
# -----------------------------
# Based on Elbow graph, K=3 works well
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# -----------------------------
# 5. Visualize Clusters
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Cluster', data=df, palette='Set2', s=100)
plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Calculate Silhouette Score
# -----------------------------
score = silhouette_score(X, df['Cluster'])
print("Silhouette Score:", round(score, 3))

# -----------------------------
# 7. Print Cluster Insights
# -----------------------------
print("\nCluster Insights:")
for cluster in sorted(df['Cluster'].unique()):
    temp = df[df['Cluster'] == cluster]
    avg_income = temp['Annual_Income'].mean()
    avg_score = temp['Spending_Score'].mean()
    print(f"Cluster {cluster}: Avg Income = ${int(avg_income)}, Avg Spending Score = {int(avg_score)}")
