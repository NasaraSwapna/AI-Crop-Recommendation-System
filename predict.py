import joblib

# Load the trained model
model = joblib.load("crop_model.pkl")

# Get input from farmer
N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorus (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter Soil pH: "))
rainfall = float(input("Enter Rainfall: "))

# Prepare input
input_data = [[N, P, K, temperature, humidity, ph, rainfall]]

# Predict crop
prediction = model.predict(input_data)

print("\nRecommended Crop:", prediction[0])