import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
import numpy as np
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.h5')
model = load_model(model_path)

# Prediction logic
def predict_action():
    try:
        stock_name = stock_entry.get()
        open_price = float(open_entry.get())
        close_price = float(close_entry.get())
        ma_diff = float(ma_entry.get())
        volume_change = float(volume_entry.get())
        rsi = float(rsi_entry.get())

        features = np.array([[open_price, close_price, ma_diff, volume_change, rsi]])
        prediction = model.predict(features)
        prediction_val = np.round(prediction[0][0], 2)
        print("Prediction value:", prediction_val)

        if prediction_val <= -0.5:
            advice = f"🔻 SELL {stock_name} — Momentum negative"
            duration = "Consider exiting within 1-2 days."
        elif prediction_val >= 0.5:
            advice = f"🟢 BUY {stock_name} — Positive signal"
            duration = "Consider holding for 3-5 days."
        else:
            advice = f"⚪ HOLD {stock_name} — No strong signal"
            duration = "Recheck after next trading session."


        result_label.config(text=f"{advice}\n{duration}", fg="#00f7ff")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

# GUI setup
root = tk.Tk()
root.title("Quantum Trade - AI Stock Trading Advisor")
root.geometry("520x520")
root.configure(bg="#121212")

# Styled label function
def styled_label(master, text):
    return tk.Label(master, text=text, font=("Segoe UI", 11), fg="#e0e0e0", bg="#121212")

# Heading
tk.Label(root, text="Quantum Trade", font=("Segoe UI", 20, "bold"), fg="#00f7ff", bg="#121212").pack(pady=20)

# Input fields
fields = [
    ("Stock Symbol (e.g., AAPL):", "stock_entry"),
    ("Open Price:", "open_entry"),
    ("Close Price:", "close_entry"),
    ("Moving Average Difference:", "ma_entry"),
    ("Volume Change (%):", "volume_entry"),
    ("RSI (Relative Strength Index):", "rsi_entry"),
]

entries = {}
for label_text, var_name in fields:
    styled_label(root, label_text).pack(pady=(8, 0))
    entry = tk.Entry(root, width=30, font=("Segoe UI", 10), bg="#1e1e2f", fg="#ffffff", insertbackground="#ffffff")
    entry.pack()
    entries[var_name] = entry

# Assign to variables
stock_entry = entries["stock_entry"]
open_entry = entries["open_entry"]
close_entry = entries["close_entry"]
ma_entry = entries["ma_entry"]
volume_entry = entries["volume_entry"]
rsi_entry = entries["rsi_entry"]

# Predict button
tk.Button(
    root, text="Get Recommendation", command=predict_action,
    bg="#00c896", fg="white", font=("Segoe UI", 12, "bold"),
    activebackground="#009f80", padx=10, pady=6
).pack(pady=20)

# Result label
result_label = tk.Label(root, text="", font=("Segoe UI", 12, "bold"), bg="#121212")
result_label.pack(pady=10)

# Start GUI loop
root.mainloop()
import joblib
scaler = joblib.load('scaler.pkl')
features = scaler.transform(np.array([[...]]))
