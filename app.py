from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from operation import LinearRegressionOperation

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    x_user = request.form['content']

    lr = LinearRegressionOperation(x_user)
    lr.parseInput()
    scores = lr.getScores()

    return render_template("review.html", scores=scores)


@app.route('/report', methods=['GET', 'POST'])
@cross_origin()
def report():
    return render_template("report.html")


if(__name__ == '__main__'):
    app.run(debug=True)
