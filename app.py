from location_mood_recommendation import food_result
from flask import Flask, render_template

app = Flask(__name__)

emotions = ["stressed", "sad", "happy", "bored"]

@app.route("/")
def index():
    return render_template("index.html", emotions=emotions);

@app.route("/emotions/<emotion>")
def handle_emotion(emotion):
    top_foods = food_result(emotion)

    food_list = ""
    for food in top_foods:
        food_list += f"<li>{food}</li>\n"
    
    return food_list;

if __name__ == '__main__':
    app.run(debug=True, port=5001)
