import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from turfpy.measurement import distance
from geojson import Point, Feature



## FOOD
food_recommend = pd.read_csv('dataset/food_coded.csv', sep=',', usecols=['comfort_food', 'comfort_food_reasons'])
food_recommend.rename(columns={'comfort_food': 'Food Types', 'comfort_food_reasons': 'Emotions'}, inplace=True)
food_recommend["Emotions"] = food_recommend["Emotions"].fillna("")
food_recommend["Food Types"] = food_recommend["Food Types"].fillna("")

stop = set(stopwords.words('english'))
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',''])
lemmatizer = WordNetLemmatizer()
food_count = {}

def preprocess_text(emotion, food_recommend):

    global food_count
    food_count = {}

    for i in range(len(food_recommend)):

        emotions_item = food_recommend["Emotions"][i]
        if isinstance(emotions_item, str):
            emotions = emotions_item.lower().split()
            emotions = [lemmatizer.lemmatize(word.strip('.,')) for word in emotions if word not in stop]

        if emotion in emotions:
            foods = food_recommend["Food Types"][i].lower().split(',')
            foods = [lemmatizer.lemmatize(food.strip().strip('.,')) for food in foods if food not in stop]

            for itemfood in foods:
                if itemfood not in food_count.keys():
                     food_count[itemfood] = 1 
                else:
                     food_count[itemfood] += 1

    top_foods = sorted(food_count, key=food_count.get, reverse=True)[:10]
    return top_foods


def food_result(emotion):
    topn = []
    topn = preprocess_text(emotion, food_recommend)
    print(f"Popular Comfort Foods in {emotion} are:")
    for food in topn:
        print(food)
    return topn[:5]

food_result('bored')

## RESTAURANT
restaurant_cusine = pd.read_csv('dataset/restaurant_location.csv', sep=',')
# print(restaurant_cusine[['categories','state']])

def get_restaurants(cusine):
    restaurants = [] 
    for i, row in cusine.iterrows():
    
        categories = row['categories']

        if categories is None:
            continue
        
         # There are nan in the categories column, which is a float
        try:
            if 'Restaurant' in categories or 'Food' in categories:
               # print("test")
                restaurants.append(row)
        except:
            continue
    return pd.DataFrame(restaurants)

restaurantsUSA_df = get_restaurants(restaurant_cusine)
# print(restaurantsUSA_df[['categories','state','longitude','latitude']])


## Kmean
kmeans = KMeans(n_clusters=6, random_state=0)
restaurantsUSA_df['cluster'] = kmeans.fit_predict(restaurantsUSA_df[['latitude', 'longitude', 'stars']])

vivid_palette = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22", "#34495e"]

plt.figure(figsize=(12, 6))
plt.axis('equal')

cluster_stats = []

for cluster_num in np.unique(restaurantsUSA_df['cluster']):
    cluster_data = restaurantsUSA_df[restaurantsUSA_df['cluster'] == cluster_num]
    plt.scatter(cluster_data['longitude'], cluster_data['latitude'], color=vivid_palette[cluster_num], label=f'Cluster {cluster_num}')

for cluster_num in np.unique(restaurantsUSA_df['cluster']):
    cluster_data = restaurantsUSA_df[restaurantsUSA_df['cluster'] == cluster_num]
    median_rating = cluster_data['stars'].median()
    center_longitude = cluster_data['longitude'].mean()
    center_latitude = cluster_data['latitude'].mean()

    plt.text(center_longitude, center_latitude, f'Median Rating: {median_rating:.2f}', fontsize=9, ha='center', va='center',
             color='black')
    
plt.grid(True)
plt.title('Location wise Restaurant Median rating in America')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster')
plt.show()


## CA
def get_restaurants(cusine, state):
    restaurants = [] 
    for i, row in cusine.iterrows():
    
        filtered_df = cusine[cusine['categories'].str.contains('Restaurant|Food', na=False)]
        filtered_df = filtered_df[filtered_df['state'].isin(['CA'])]
        return filtered_df

restaurants_in_ca = get_restaurants(restaurant_cusine,'CA')
print(restaurants_in_ca[['categories','state']])


rest_loc = restaurants_in_ca[['name','longitude','latitude','stars','categories','state','address','city','postal_code']]
total_restaurants = len(rest_loc)
# print(f"Number of total restaurant in CA is {total_restaurants}")

# UCLA coordinates
ucla_loc = Feature(geometry=Point((-118.4452, 34.0689))) 

distance_threshold = 105
count_restaurant = 0
restaurants_near_ucla = []

for i, row in rest_loc.iterrows():
    restaurant_point = Feature(geometry=Point((row['longitude'], row['latitude'])))
    dist = distance(ucla_loc, restaurant_point, units='km')
    if dist <= distance_threshold:
        count_restaurant += 1
        restaurants_near_ucla.append(row)

print(f"Number of closest restaurants within {distance_threshold} km of UCLA: {count_restaurant}")
print("=======================================================")
df_near_ucla = pd.DataFrame(restaurants_near_ucla)
df_near_ucla.rename(columns={'name': 'Name', 'stars': 'Rating', 'categories': 'Food Category'}, inplace=True)
print(df_near_ucla[['Name','Rating','Food Category']])


## RECOMMENDATION
mean_rating = df_near_ucla['Rating'].mean()
emotions_and_foods = {
    "stressed": ["burger", "cafes", "donuts", "ice cream", "chip", "pasta", "french fries"],
    "happy": ["pizza", "ice cream", "sub", "chicken wings", "pretzel", "hot dogs", "deli sandwich"],
    "sadness": ["ice cream", "pizza", "deli", "mac and cheese", "pretzel", "chinese", "pasta", "burger"],
    "bored": ["sandwiches", "cookie", "vietnamese", "mac and cheese", "chicken wings"]
}

def get_restaurant_recommendations():
    df_all_recommendations = pd.DataFrame()
    for emotion, comfort_foods in emotions_and_foods.items():
        for food in comfort_foods:
            filtered_restaurants = df_near_ucla[
            (df_near_ucla['Food Category'].str.contains(food, case=False)) &
                (df_near_ucla['Rating'] >= df_near_ucla['Rating'].mean())
            ]
            top_recommendations = filtered_restaurants.sort_values(by='Rating', ascending=False)
            if not top_recommendations.empty:
                top_recommendations['Comfort Food'] = food.title()
                top_recommendations['Emotion'] = emotion
                df_all_recommendations = pd.concat([df_all_recommendations, top_recommendations])
    return df_all_recommendations[['Emotion', 'Comfort Food', 'Name', 'Rating', 'address', 'city', 'state', 'postal_code']]
