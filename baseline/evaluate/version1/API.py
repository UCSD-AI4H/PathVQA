from flask import Flask, request, render_template
import flask_restful as restful
from flask_restful import reqparse
from evaluator import *
import re
app = Flask(__name__)
api = restful.Api(app)
parser_put = reqparse.RequestParser()
parser_put.add_argument("standard_output", type=str, required=True, help="needs standard output whose type is str and which contains at least one letter")
parser_put.add_argument("predict_output", type=str, action='append', required=True, help="needs predict output whose type is list of str")
parser_put.add_argument("n", type=int, required=True, help="needs n whose type is int, used for BLEU-n")
parser_put.add_argument("weights", type=str, action='append', required=True, help="needs weights whose type is list of float, used for the weight of BLEU-n")

class similarity(restful.Resource):

    def post(self):
        args = parser_put.parse_args()
        candidate = args["standard_output"]
        references = args["predict_output"]
        n = args["n"]
        weights = args["weights"]

        return {'bleu': str(evaluator(candidate, references, n, weights))}, 201

api.add_resource(similarity, '/similarity')

if __name__ == "__main__":
    app.run(debug=True)
