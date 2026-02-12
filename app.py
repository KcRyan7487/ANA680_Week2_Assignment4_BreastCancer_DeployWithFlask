from flask import Flask, request, jsonify, render_template, redirect, url_for
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from pathlib import Path


# Create Flask app
app = Flask(__name__)


# Initialize and train a simple regression model (toy example)
# model = LinearRegression()

# Sample training data (X: input, y: output)
# X_train = np.array([[1], [2], [3], [4], [5]])
# y_train = np.array([1, 2, 3, 3.5, 4.5])

# # Train the model
# model.fit(X_train, y_train)


# Route for homepage
#@app.route('/')
#def hello():
#    return 'Hello, World!'

#@app.route("/")
#def hello():
#    return """
#    <h1>Hello, World!</h1>
#
#    <p>This is the home page.</p>
#
#    <a href="/predict?value=2">
#        Click here to run prediction example (value = 2)
#    </a>
#    """
#

# Load in the pickle file of our naive bayes model (which is the one which "won" last week week 1... 
# per the assignment instructions we'll use this one again this week but serve its predictions based on the HTML inputs as outputs to the user)
MODEL_PATH = Path("model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# List of the columns/features (besides the target actual class)
FEATURE_ORDER = [
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses"
]

# Set up home page similarly as before
@app.route("/")
def home():
    return render_template("index.html")


# Still supports going to /predict this time but redirect to home now instead of hello before
# The real functionality here though is the many multiple input fields now in the form and how those feed into the model inputs so that it can output the prediction for benign (2) or malignant (4)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # If someone types /predict in the address bar, bounce them to home
    if request.method == "GET":
        return redirect(url_for("home"))

    # Read the 9 inputs from the HTML form, in correct order
    values = []
    for feature in FEATURE_ORDER:
        raw = request.form.get(feature, "").strip()
        values.append(float(raw))

    X_input = np.array(values).reshape(1, -1)

    pred = model.predict(X_input)[0]

    # Class mapping: 2 = benign, 4 = malignant
    if int(pred) == 2:
        result = "Benign (Class = 2)"
    elif int(pred) == 4:
        result = "Malignant (Class = 4)"
    else:
        result = f"Unknown class returned: {pred}"

    return render_template(
        "index.html",
        result=result
    )


# I can see in Anaconda Prompt that it throws a 404 error on the browser hitting our local site when it does the initail ask for the favicon which is just the web icon that shows on the tab for the website. 
# It also appears yello in the log due to registering as a 404 error.... but if we add an entry here for a 204 explicitly stating there is no content it should stop registering as an erorr 
@app.route('/favicon.ico')
def favicon():
    return ('', 204)


#Another new route now for the alternative more complicated page/setup for deploying the linear regression.... user must put the right URL in with a value to output result

#@app.route("/predict", methods=["GET"])
#def predict():
#    # Get query parameter ?value=...
#    input_value = request.args.get("value", type=float)
#
#    if input_value is None:
#        return jsonify({"error": "Please provide a numeric value using ?value=<number>"}), 400
#
#    prediction = model.predict(np.array([[input_value]]))[0]
#
#    return jsonify({
#        "input": input_value,
#        "prediction": float(prediction)
#    })


#Could implement basic navigation between pages I guess using this functionality then.... something like these 2 now linking to each other back and forth
#@app.route("/predict", methods=["GET"])
#def predict():
#
#    input_value = request.args.get("value", type=float)
#
#    if input_value is None:
#        return jsonify({"error": "Please provide ?value=<number>"}), 400
#
#    prediction = model.predict(np.array([[input_value]]))[0]
#
#    return f"""
#    <h2>Prediction Result</h2>
#
#    <p><b>Input:</b> {input_value}</p>
#    <p><b>Prediction:</b> {prediction:.10f}</p>
#
#    <br>
#    <a href="/">Go back to Home</a>
#    """
#

#Next stepup now doing it from the html instead of only from this app.py:
#@app.route("/predict", methods=["GET", "POST"])
#def predict():
#    #This will just also still allow a navigate straight to the URL like before instead of erroring but now it'll re-direct back to the home for the form input
#    if request.method == "GET":
#        return redirect(url_for("hello"))
#
#    input_value = float(request.form["input_value"])
#    prediction = model.predict(np.array([[input_value]]))[0]
#
#    return render_template(
#        "index.html",
#        input_value=input_value,
#        prediction=prediction
#    )
#
#



# Run locally
if __name__ == '__main__':
    app.run(debug=True)



