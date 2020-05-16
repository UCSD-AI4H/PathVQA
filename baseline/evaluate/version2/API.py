from flask import Flask, request, render_template
import flask_restful as restful
from flask_restful import reqparse
from evaluator import *
import re
app = Flask (__name__)
@app.route('/search4',methods=['POST'])
def do_search() -> str:
    candidate = request.form['candidate']
    reference = request.form['reference']
    n = request.form['n']
    n = int(n)
    weights = request.form["weights"]
    weights = re.findall(r'-?\d+\.?\d*e?-?\d*?', weights)
    result = str(evaluator(candidate, [reference], n, weights))
    title = "Here are your results:"
    return render_template('result.html',
                           the_title = title,
                           the_standard_output = candidate,
                           the_predict_output = reference,
                           the_n = n,
                           the_weights = weights,
                           the_result = result,)

@app.route("/")
@app.route('/entry')
def entry_page() -> 'html':
    return render_template('entry.html',
                           the_title='Welcome to BLEU-n on the web!')
if __name__ == "__main__":
    app.run(debug=True)

