from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("crop_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]

        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)