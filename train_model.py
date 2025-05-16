# train_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

# Generate dummy financial data
np.random.seed(42)
data = pd.DataFrame({
    'open': np.random.rand(1000),
    'high': np.random.rand(1000),
    'low': np.random.rand(1000),
    'close': np.random.rand(1000),
    'volume': np.random.rand(1000),
})
data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
data.dropna(inplace=True)

# Features and target
X = data[['open', 'high', 'low', 'close', 'volume']]
y = data['target']

# Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(64, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Save the model
model.save('model.h5')

print("✅ Model saved as model.h5")
