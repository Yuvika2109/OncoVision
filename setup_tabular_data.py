from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Create folder
os.makedirs('data/tabular', exist_ok=True)

# Download and save dataset
print("📥 Downloading Wisconsin Breast Cancer Dataset...")
data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

df.to_csv('data/tabular/wbcd.csv', index=False)

print("✅ Dataset saved to: data/tabular/wbcd.csv")
print(f"📊 Total samples: {len(df)}")
print(f"📈 Total features: {len(data.feature_names)}")
print(f"🎯 Classes: Malignant (0) = {sum(df['target']==0)}, Benign (1) = {sum(df['target']==1)}")