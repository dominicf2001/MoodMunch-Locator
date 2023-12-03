from location_mood_recommendation import food_result
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    print("Hello world")
    return ""
