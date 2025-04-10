from flask import Flask, request, jsonify, send_file, render_template
import re
from io import BytesIO, StringIO  # Import StringIO
import nltk
# nltk.download('stopwords')  # This is only needed once, not on every run.
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
from flask_cors import CORS


STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)

# Load the models and vectorizer *once* when the app starts.  VERY IMPORTANT.
predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))


@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == '':   # added the file name check
                return jsonify({"error": "No selected file"})
            data = pd.read_csv(file,  delimiter = '\t', quoting = 3) # added delimiter and quotechar
            predictions, graph = bulk_prediction(data)
            # Create a CSV in-memory buffer
            csv_buffer = StringIO()
            data.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            response = send_file(
                BytesIO(csv_buffer.getvalue().encode()),
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"
            response.headers["X-Graph-Data"] = base64.b64encode(graph.getvalue()).decode("ascii")
            return response

        elif "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        print(f"Error in /predict: {e}")  # More detailed error logging
        return jsonify({"error": str(e)})


def single_prediction(text_input):
    # Use the PRE-TRAINED cv and scaler objects.
    global predictor, scaler, cv  # Access the global variables

    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    
    # Transform the SINGLE review using the loaded CountVectorizer.
    X_prediction = cv.transform([review]).toarray()  # Pass review as a list
    X_prediction_scl = scaler.transform(X_prediction)
    print("------ Debugging: single_prediction ------")
    print("Original text_input:", text_input)
    print("Processed review:", review)
    print("CountVectorizer feature names (first 50):", cv.get_feature_names_out()[:50])  # Show first 50 features
    print("Transformed X_prediction:", X_prediction)
    print("Scaled X_prediction_scl:", X_prediction_scl)
    print("------------------------------------------")
    y_predictions = predictor.predict_proba(X_prediction_scl)

    print("Raw Prediction Probabilities:", y_predictions)  # Debugging Line

    y_predictions = y_predictions.argmax(axis=1)[0]
    print("Final Prediction:", y_predictions)  # Debugging Line

    return "Positive" if y_predictions == 1 else "Negative"



def bulk_prediction(data):
    global predictor, scaler, cv  # Access the global variables
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["verified_reviews"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    # predictions_csv = BytesIO()  # No longer needed here

    # data.to_csv(predictions_csv, index=False) # No longer needed
    # predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return  data, graph  # Return data directly


def get_distribution_graph(data):
   # Create Matplotlib plot
    fig = plt.figure(figsize=(7, 7))  # set fig size
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.1, 0.1)
    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )


    # Convert plot to bytes
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close(fig)  # Close the figure to free memory
    img_buffer.seek(0)  # Rewind the buffer to the beginning

    return img_buffer



def sentiment_mapping(x):
    if x == 1:
        return "Positive"
    else:
        return "Negative"


if __name__ == "__main__":
    app.run(port=5000, debug=True)