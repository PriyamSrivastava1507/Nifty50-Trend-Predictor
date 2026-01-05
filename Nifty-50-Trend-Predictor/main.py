import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv('content/nifty.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

for col in ['Open', 'High', 'Low', 'Close']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

df['Return_1d'] = df['Close'].pct_change()
df['Return_5d'] = df['Close'].pct_change(5)

df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
df = df.dropna().reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], color='blue', linewidth=1)
plt.title('Nifty 50 Historical Close Price (1995-2024)', fontsize=14)
plt.ylabel('Price (INR)')
plt.xlabel('Year')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('nifty_price_trend.png')
plt.show()

features = ['RSI', 'Return_1d', 'Return_5d', 'SMA_10', 'SMA_50']
X = df[features]
Y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, Y_train, Y_valid = train_test_split(X_scaled, Y, test_size=0.1, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train, Y_train)

print("Improved Classification Report:")
print(metrics.classification_report(Y_valid, model.predict(X_valid)))

metrics.ConfusionMatrixDisplay.from_estimator(model, X_valid, Y_valid, cmap='RdYlGn')
plt.title('Improved Model: Random Forest with Technical Indicators')
plt.savefig('nifty_confusion_matrix.png')
plt.show()