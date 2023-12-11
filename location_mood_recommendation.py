import pandas as pd
import numpy as np
import json
import random

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# FOOD LOGIC
# ----------
food_recommend = pd.read_csv('dataset/food_coded.csv', sep=',', usecols=['comfort_food', 'comfort_food_reasons'])

# only take two attributes comfort_food and comfort_food_reason and rename
food_recommend.rename(columns={'comfort_food': 'Food Types', 'comfort_food_reasons': 'Emotions'}, inplace=True)

food_recommend["Emotions"] = food_recommend["Emotions"].fillna("")
food_recommend["Food Types"] = food_recommend["Food Types"].fillna("")

# Filter all common words
stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',''])
lemmatizer = WordNetLemmatizer()
food_count = {}

"""
    Finds the top comfort foods associated with a given mood.

    emotion (str): The emotion to search for.
    food_recommend (DataFrame): DataFrame containing 'Food Types' and 'Emotions' columns.

    Returns:
    list: A list of the top comfort foods for the given mood.
    
"""

# Processing
def preprocess_text(emotion, food_recommend):

    global food_count
    food_count = {}

    # Looping through the food data
    for i in range(len(food_recommend)):

        # PROCESS "comfort_food_reasons"
        emotions_item = food_recommend["Emotions"][i]
        # Convert all items in comfort_food_reasons to str included NaN value.
        # Split it into individual words, removes punctuation (. ,) and converts to lowercase
        # checks if each word is not a stop word. (and with NLTK, common words will be removed such as "I","and")
        emotions = []
        if isinstance(emotions_item, str):
            emotions = emotions_item.lower().split()
            emotions = [lemmatizer.lemmatize(word.strip('.,')) for word in emotions if word not in stop]

        # PROCESS "comfort_food"
        # If the mood is found, the processed similarly: split into item, punctuation removed, converted to lowercase, and lemmatized
        if emotion in emotions:
            foods = food_recommend["Food Types"][i].lower().split(',')
            foods = [lemmatizer.lemmatize(food.strip().strip('.,')) for food in foods if food not in stop]

        # Add process food to food count and count food
        # If the item is new to the dictionary, added with a count of 1; if it already exists, its count is incremented
            for itemfood in foods:
                if itemfood not in food_count.keys():
                     food_count[itemfood] = 1 
                else:
                     food_count[itemfood] += 1

    # Now specified mood is already associated with food.
    # Sorting and selecting the top foods (most to least appearing food)
    top_foods = sorted(food_count, key=food_count.get, reverse=True)
    return top_foods


def food_result(emotion):
    topn = []
    topn = preprocess_text(emotion, food_recommend) #function create dictionary only for particular mood
    return topn[:5]
# ----------

def kmeans(data, k):
    centroids  = []
    for _ in range(k):
        i = random.randint(0, len(data) - 1)
        centroids.append(data[i])

    
    return

# RESTAURANT LOGIC
# ----------
def get_restaurants(businesses):
    i = 0
    restaurants = []
    for business in businesses:
        if i == 100:
            break
        
        categories = business['categories']

        if categories is None:
            continue
        
        is_restaurant = 'Restaurants' in categories or 'Food' in categories
        if is_restaurant:
            restaurants.append(business)
        i += 1
        
    return restaurants

businesses = []
with open('dataset/yelp_businesses.json', 'r') as file:
    for line in file:
        businesses.append(json.loads(line))

restaurants = get_restaurants(businesses)
kmeans(restaurants, 4)

# ----------

