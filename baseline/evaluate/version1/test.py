import requests
import json
url = "http://127.0.0.1:5000/similarity"
parms = {
    "standard_output":"It is a guide to action which ensures that the military always obeys the commands of the party",
    "predict_output": ["It is a guide to action that ensures that the military will forever heed Party commands",
                  "It is the guiding principle which guarantees the military forces always being under the command of the Party",
                  "It is the practical guide for the army always to heed the directions of the party"],
    "n": 3,
    "weights": [0.25, 0.25, 0.25, 0.25]
}
result = requests.post(url, data=parms)
text = result.text
print(json.loads(text))
