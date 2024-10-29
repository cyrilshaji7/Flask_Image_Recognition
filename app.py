from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiate the Flask app
app = Flask(__name__)

@app.route("/")
def main():
    """Render the main page."""
    return render_template("index.html")

@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """Handle the prediction of an image file."""
    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except Exception as e:
        # Handle specific exceptions here if needed
        error = "File cannot be processed."
        return render_template("result.html", err=error)

# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
