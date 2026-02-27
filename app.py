from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    msg = request.form["message"]

    data = vectorizer.transform([msg])

    result = model.predict(data)[0]

    if result == 1:
        output = "Spam"
    else:
        output = "Not Spam"

    return render_template("index.html", prediction=output)

app.run(debug=True)