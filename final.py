from pyspark import SparkContext
import sys
import time
import json
from datetime import datetime
import joblib
import numpy as np
from xgboost import XGBRegressor
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder


folder_path = sys.argv[1]
output_file_path = sys.argv[3]
validation_file_path = sys.argv[2]

sc = SparkContext('local[*]', 'model_base_rec')
sc.setLogLevel("ERROR")

start = time.time()

yelp_data = '/yelp_train.csv'
yelp_train_data_rdd = sc.textFile(folder_path + yelp_data)
row_one = yelp_train_data_rdd.first()
yelp_train_data_rdd = yelp_train_data_rdd.filter(lambda a: a != row_one).map(lambda a: a.split(','))

# club
# yelp_data_1 = '/yelp_val.csv'
# yelp_train_data_rdd_1 = sc.textFile(folder_path + yelp_data_1)
# row_one_1 = yelp_train_data_rdd_1.first()
# yelp_train_data_rdd_1 = yelp_train_data_rdd_1.filter(lambda a: a != row_one_1).map(lambda a: a.split(','))
#
# combined_rdd = yelp_train_data_rdd.union(yelp_train_data_rdd_1)
# club

current_year = sc.broadcast(datetime.now().year)
user_file = '/user.json'
user_doc = sc.textFile(folder_path + user_file)

def process_user_data(user):
    user_id = user["user_id"]
    review_count = float(user["review_count"])
    yelping_since = current_year.value - int(user["yelping_since"][:4])
    average_stars = float(user["average_stars"])
    useful = int(user["useful"])
    funny = int(user["funny"])
    cool = int(user["cool"])
    fans = int(user["fans"])
    friends = user.get("friends", "None")
    friends_count = len(friends.split(', ')) if friends != "None" else 0

    compliments_sum = sum([int(user[key]) for key in user.keys() if key.startswith('compliment_')])


    return (user_id, (review_count, yelping_since, average_stars, useful, funny, cool, fans, friends_count, compliments_sum))

user_features_rdd = user_doc.map(lambda u: json.loads(u)).map(process_user_data)
uz_features = dict(user_features_rdd.collect())

biz_file = '/business.json'
biz_doc = sc.textFile(folder_path + biz_file)


def process_biz_data(biz):
    biz_id = biz['business_id']
    stars = float(biz['stars'])
    biz_review_count = float(biz['review_count'])
    is_open = biz['is_open']
    categories = biz.get('categories', '')
    if categories:
        categories_list = categories.split(', ')
        num_categories = len(categories_list)
    else:
        num_categories = 0

    state = biz.get('state', '')
    label_encoder = LabelEncoder()
    encoded_state = label_encoder.fit_transform([state])[0]
    latitude = biz.get('latitude', 0.0)
    longitude = biz.get('longitude', 0.0)

    hours = biz.get('hours')
    if hours:
        total_hours_open = 0
        for day, hours_range in hours.items():
            open_time, close_time = map(int, hours_range.split('-')[0].split(':')), map(int,hours_range.split('-')[1].split(':'))
            open_hours = next(open_time)
            open_minutes = next(open_time)
            close_hours = next(close_time)
            close_minutes = next(close_time)
            total_hours_open += (close_hours - open_hours) + ((close_minutes - open_minutes) / 60)
    else:
        total_hours_open = 0

    return (biz_id, (stars, biz_review_count, is_open, num_categories, encoded_state, latitude, longitude, total_hours_open))

biz_features_rdd = biz_doc.map(lambda b: json.loads(b)).map(process_biz_data)
biz_features = dict(biz_features_rdd.collect())

# tips

tips_file = '/tip.json'
tip_doc = sc.textFile(folder_path + tips_file)

def process_tip_data(tip):
    biz_id = tip['business_id']
    tip_likes_count = float(tip['likes'])
    return (biz_id, (tip_likes_count))

tip_features_rdd = tip_doc.map(lambda b: json.loads(b)).map(process_tip_data)
tip_features = dict(tip_features_rdd.collect())

# tips

#photos
photo_file = '/photo.json'
photo_doc = sc.textFile(folder_path + photo_file)

def process_photo_data(photo):
    biz_id = photo['business_id']
    photo_id = photo['photo_id']
    return (biz_id, photo_id)

photo_features_rdd = photo_doc.map(lambda p: json.loads(p)).map(process_photo_data)
distinct_photo_counts = photo_features_rdd.groupByKey().mapValues(lambda x: len(set(x)))
distinct_photo_counts_dict = dict(distinct_photo_counts.collect())
# print('distinct photos: ', distinct_photo_counts_dict)

#photos

def extract_features(record):
    user, biz, rating = record
    if user in uz_features:
        review_count, yelping_since, average_stars, useful, funny, cool, fans, friends_count, compliments_sum = uz_features[user]
    else:
        review_count, yelping_since, average_stars, useful, funny, cool, fans, friends_count, compliments_sum = None, None, None, None, None, None, None, None, None

    if biz in biz_features:
        stars, biz_review_count, is_open, num_categories, encoded_state, latitude, longitude, total_hours_open = biz_features[biz]
    else:
        stars, biz_review_count, is_open, num_categories, encoded_state, latitude, longitude, total_hours_open = None, None, None, None, None, None, None, None

    if biz in tip_features:
        tip_likes_count = tip_features[biz]
    else:
        tip_likes_count = None

    if biz in distinct_photo_counts_dict:
        photo_count = distinct_photo_counts_dict[biz]
    else:
        photo_count = None

    return (review_count, yelping_since, average_stars, useful, funny, cool, fans, friends_count, compliments_sum, stars, biz_review_count, is_open, num_categories, encoded_state, latitude, longitude, total_hours_open, tip_likes_count, photo_count, rating)


X_train_and_Y_train = yelp_train_data_rdd.map(extract_features)
X_train = X_train_and_Y_train.map(lambda a: a[:-1]).collect()
Y_train = X_train_and_Y_train.map(lambda a: a[-1]).collect()

yelp_val_data_rdd = sc.textFile(validation_file_path)
row_one_val = yelp_val_data_rdd.first()
yelp_val_data_rdd = yelp_val_data_rdd.filter(lambda a:a != row_one_val).map(lambda a: a.split(","))
yelp_val_uz_biz = yelp_val_data_rdd.map(lambda a: (a[0], a[1]))
uz_biz = yelp_val_uz_biz.collect()
#print('1: ',yelp_val_uz_biz.take(10))

def extract_validation_features(record):
    user, biz = record
    if user in uz_features:
        review_count, yelping_since, average_stars, useful, funny, cool, fans, friends_count, compliments_sum = uz_features[user]
    else:
        review_count, yelping_since, average_stars, useful, funny, cool, fans, friends_count, compliments_sum = None, None, None, None, None, None, None, None, None

    if biz in biz_features:
        stars, biz_review_count, is_open, num_categories, encoded_state, latitude, longitude, total_hours_open = biz_features[biz]
    else:
        stars, biz_review_count, is_open, num_categories, encoded_state, latitude, longitude, total_hours_open = None, None, None, None, None, None, None, None

    if biz in tip_features:
        tip_likes_count = tip_features[biz]
    else:
        tip_likes_count = None

    if biz in distinct_photo_counts_dict:
        photo_count = distinct_photo_counts_dict[biz]
    else:
        photo_count = None

    return (review_count, yelping_since, average_stars, useful, funny, cool, fans, friends_count, compliments_sum, stars, biz_review_count, is_open, num_categories, encoded_state, latitude, longitude, total_hours_open, tip_likes_count, photo_count)

validation_dataset = yelp_val_uz_biz.map(extract_validation_features)

X_val = validation_dataset.map(lambda a: a).collect()

X_train = np.array(X_train, dtype='float32')
Y_train = np.array(Y_train, dtype='float32')
X_val = np.array(X_val, dtype='float32')


# tuning
# param_grid = {
#     'learning_rate': [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.45],
#     'max_depth': [10, 15, 17, 18, 19, 20, 25],
#     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
#     'n_estimators': [450, 500, 700, 900, 950, 1000],
#     'alpha': [1, 7, 14, 18, 23],
#     'reg_lambda': [45, 50, 65, 70, 75, 80],
#     'min_child_weight': [120, 240, 360, 420]
# }
# xgb = XGBRegressor(random_state=1234)
# random_search = RandomizedSearchCV(
#     xgb,
#     param_distributions=param_grid,
#     n_iter=25,
#     scoring='neg_root_mean_squared_error',
#     cv=5,
#     verbose=2,
#     random_state=1234,
#     n_jobs=-1
# )
# random_search.fit(X_train, Y_train)
# best_params = random_search.best_params_
# print("Best Parameters:", best_params)
# best_model = XGBRegressor(random_state=1234, **best_params)
# best_model.fit(X_train, Y_train)
# predicted_val = best_model.predict(X_val)
# with open('best_params_new.pkl', 'wb') as file:
#     pickle.dump(best_params, file)
# tuning

#
best_model = XGBRegressor(
    learning_rate=0.02,
    colsample_bytree=0.6,
    max_depth=15,
    subsample=0.8,
    n_estimators=950,
    alpha=18,
    reg_lambda=80,
    min_child_weight=240
    )

# with open('best_params.pkl', 'rb') as file:
#     best_params = pickle.load(file)
#
# best_model = XGBRegressor(**best_params)
best_model.fit(X_train, Y_train)
# with open('best_model_v2.pkl', 'wb') as file:
#     pickle.dump(best_model, file)

# joblib.dump(best_model, 'best_model_v2.joblib', compress=('zlib', 9))
# loaded_model = joblib.load('best_model_v2.joblib')
# with open('best_model_v2.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

predicted_val = best_model.predict(X_val)

heading = "user_id, business_id, prediction\n"
with open(output_file_path, "w") as output_file:
    output_file.write(heading)
    for i in range(0, len(predicted_val)):
        output_file.write(f'{uz_biz[i][0]},{uz_biz[i][1]},{str(predicted_val[i])}\n')
end = time.time()
print('Duration: ', end - start)
sc.stop()