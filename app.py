"""
app.py: Main application for image prediction using Flask.
"""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiate Flask app
app = Flask(__name__)

# Home route
@app.route("/")
def main():
    return render_template("index.html")

# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """
    Process the uploaded image file and return predictions.
    """
    try:
        if request.method == 'POST':
            img = preprocess_img(request.files['file'].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))
    except FileNotFoundError:  # Catch specific exceptions
        error = "File cannot be processed."
        return render_template("result.html", err=error)
    except Exception:  # Catch other general exceptions
        error = "An unexpected error occurred."
        return render_template("result.html", err=error)

# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
