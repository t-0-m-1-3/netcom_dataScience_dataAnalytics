import pyspark
from flask import Flask, request, jsonify
from pyspark.ml import PipelineModel

app = Flask(__name__)

MODEL=pyspark.ml.PipelineModel("spark-naive-bayes-model")

HTTP_BAD_REQUEST = 400

@app.route('/predict')
def predict():
    Description = request.args.get('Description', default=None, type=str)

    # Reject request that have bad or missing values.
    if Description is None:
        # Provide the caller with feedback on why the record is unscorable.
        message = ('Record cannot be scored because of '
                   'missing or unacceptable values. '
                   'All values must be present and of type string.')
        response = jsonify(status='error',
                           error_message=message)
        # Sets the status code to 400
        response.status_code = HTTP_BAD_REQUEST
        return response

    features = [[Description]]
    predictions = MODEL.transform(features)
    label_pred = predictions.select("Description","Category","probability","prediction")
    return jsonify(status='complete', label=label_pred)

if __name__ == '__main__':
    app.run(debug=True,port=4444)
