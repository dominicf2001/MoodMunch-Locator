from prompt import food_result, get_restaurant_recommendations
from flask import Flask, render_template

app = Flask(__name__)

emotions = ["stressed", "sad", "happy", "bored"]

@app.route("/")
def index():
    return render_template("index.html", emotions=emotions);

@app.route("/emotions/<emotion>")
def handle_emotion(emotion):
    top_foods = food_result(emotion)
    restaurants = get_restaurant_recommendations(emotion)
    
    food_list = ""
    for food in top_foods:
        food_list += f"<li>{food}</li>\n"

    restaurant_list = '<ul class="results" id="restaurantResultsList" hx-swap-oob="innerHTML">\n'
    restaurants.sort_values(by='Rating')
    restaurants = restaurants[0:5]
    for i, restaurant in restaurants.iterrows():
        name = restaurant["Name"]
        rating = restaurant["Rating"]
        address = restaurant["address"]
        restaurant_list += f'<li>{name}: {address} ({rating} stars)</li>\n'
    restaurant_list += '</ul>'
     
    return food_list + restaurant_list

if __name__ == '__main__':
    app.run(debug=True, port=5001)
