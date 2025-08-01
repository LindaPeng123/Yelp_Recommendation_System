'''
Method Description:
In this project, I use a hybrid recommendation system combining item-based with model-based recommendation systems to predict users' ratings for the Yelp dataset. I first designed an item-based collaborative filtering recommendation system with Pearson similarity. I have used default voting to shrink item similarities based on the number of common users with the active item. For the model-based one, I use the XGBoost regressor to predict the score given by a user to a business based on their features. I have created 42 features from datasets of users, business, photo, checkin, and tip. Some unexpected features with large improvements of this model include the number of friends for each user, the timestamp for a user joining Yelp, the number of categories for each business, whether the business has outdoor seating or not, the length and timestamp of tips, the total number of check-ins, and the number of photo captions. I also imputed missing values in numerical features with either mean or median based on their characteristics. I also assigned the average value for creating new user or business features that do not exist in the training set. To improve the model's accuracy, I have used the grid search to tune hyperparameters and parameters like gamma, min_child_weight, and max_depth to prevent overfitting. After obtaining predicted scores of the validation set from both recommendation systems, I restricted each score's range from 1 to 5. Then, I applied linear regression to calculate weights for combining these two scores using the weighted average. The exact weights will be directly applied to combining the training and test set without re-conducting the linear regression. Finally, I restricted the combined scores again from 1 to 5 to improve the accuracy. 

Error Description:
>=0 and <1: 102668
>=1 and <2: 32455
>=2 and <3: 6113
>=3 and <4: 808
>=4: 0

RMSE:
0.9749191634715715

Execution Time:
750.868451833725s
'''

# %%
from pyspark import SparkContext
import sys
import os
import numpy as np
from xgboost import XGBRegressor
import json
import statistics as stat
from datetime import datetime
import math
import time

# %%
sc = SparkContext('local[*]', 'competition')
sc.setLogLevel('WARN')

# %%
folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]

def prediction_calculation(user, bus, bus_user_rating, user_avg_rating_dict, bus_avg_rating_dict, user_bus_dict, bus_user_dict, shrink_factor, top_num):
    bus_bus1_sim = {}
    if bus not in bus_user_rating.keys() and user not in user_avg_rating_dict.keys():
        return (user, bus, 3)
    elif bus not in bus_user_rating.keys():
        return (user, bus, user_avg_rating_dict[user])
    elif user not in user_avg_rating_dict.keys():
        return (user, bus, bus_avg_rating_dict[bus])
    else:
        user_list = bus_user_dict[bus]
        if user in user_list:
            return (user, bus, bus_user_rating[bus][user])
        else:
            bus_list = user_bus_dict[user]
            pearson_sim_list_temp = []
            active_user_rating_list = []
            for bus1 in bus_list:
                if bus1 == bus:
                    continue
                active_user_rating = bus_user_rating[bus1][user]
                active_user_rating_list.append(active_user_rating)
                if (bus1, bus) in bus_bus1_sim:
                    pearson_sim_list_temp.append(bus_bus1_sim[(bus1, bus)])
                    continue
                user_list_bus1 = bus_user_dict[bus1]
                common_user = user_list.intersection(user_list_bus1)
                if len(common_user) <= 2:
                    pearson_sim = 1 - (abs(bus_avg_rating_dict[bus] - bus_avg_rating_dict[bus1]) / 5)
                    pearson_sim_list_temp.append(pearson_sim)
                    continue
                common_user_rating = [(i, bus_user_rating[bus][i], bus_user_rating[bus1][i]) for i in common_user]
                bus_co_avg = stat.mean([i[1] for i in common_user_rating])
                bus1_co_avg = stat.mean([i[2] for i in common_user_rating])
                minus_list = [(i[1] - bus_co_avg, i[2] - bus1_co_avg) for i in common_user_rating]
                numerator = sum([i[0] * i[1] for i in minus_list])
                if numerator == 0:
                    pearson_sim_list_temp.append(0)
                    continue
                denominator = math.sqrt(sum(i[0] ** 2 for i in minus_list)) * math.sqrt(sum(i[1] ** 2 for i in minus_list))
                temp_sim = numerator / denominator
                common_user_num = max(0.1, len(common_user))
                pearson_sim = (common_user_num / (common_user_num + shrink_factor)) * temp_sim
                bus_bus1_sim[(bus, bus1)] = pearson_sim
                pearson_sim_list_temp.append(pearson_sim)
            pearson_sim_list = [0.1] * len(pearson_sim_list_temp) if sum(pearson_sim_list_temp) == 0 else pearson_sim_list_temp
            similarity_rating = sorted([(i, j)for i, j in zip(pearson_sim_list, active_user_rating_list) if i != 0], key = lambda x: x[0], reverse = True)
            similarity_rating_final = similarity_rating[:top_num] if len(similarity_rating) > top_num else similarity_rating
            prediction = max(sum(i[0] * i[1] for i in similarity_rating_final) / sum((abs(i[0]) for i in similarity_rating_final)), 1)
            return (user, bus, prediction)


# %%
def item_based_cf(train_rdd, val_rdd):
    bus_user_rating_list = train_rdd.groupBy(lambda x: x[1]).mapValues(lambda x: {i[0]: float(i[2]) for i in x}).collect()
    bus_user_rating = dict(bus_user_rating_list)
    bus_user_dict = {bus: set(bus_user_rating[bus].keys()) for bus in bus_user_rating}

    user_avg_rating = train_rdd.groupBy(lambda x: x[0]).mapValues(lambda x: stat.mean([float(i[2]) for i in x])).collect()
    bus_avg_rating = train_rdd.groupBy(lambda x: x[1]).mapValues(lambda x: stat.mean([float(i[2]) for i in x])).collect()
    user_avg_rating_dict = dict(user_avg_rating)
    bus_avg_rating_dict = dict(bus_avg_rating)

    user_bus = train_rdd.groupBy(lambda x: x[0]).mapValues(lambda x: [i[1] for i in x]).collect()
    user_bus_dict = dict(user_bus)

    prediction_rdd = val_rdd.map(lambda x: prediction_calculation(x[0], x[1], bus_user_rating, user_avg_rating_dict, bus_avg_rating_dict, user_bus_dict, bus_user_dict, 11, 8))
    prediction_list = prediction_rdd.collect()
        
    return prediction_list

# %%
def create_feature(user, bus, user_data_dict, bus_data_dict, user_data_avg, bus_data_avg, tip_dict, user_tip_type_avg, bus_tip_type_avg, tip_type_avg, tip_user, tip_bus):
    if user in user_data_dict.keys():
        features_list = list(user_data_dict[user])
    else:
        features_list = user_data_avg

    if bus in bus_data_dict.keys():
        features_list += list(bus_data_dict[bus])
    else:
        features_list += bus_data_avg + [None] * 13
    
    if (user, bus) in tip_dict.keys():
        features_list += tip_dict[(user, bus)]
    elif (user in tip_user) and (bus in tip_bus):
        features_list += list(np.mean([user_tip_type_avg[user], bus_tip_type_avg[bus]], axis = 0))
    elif (user not in tip_user) and (bus not in tip_bus):
        features_list += tip_type_avg
    elif (user not in tip_user) and (bus in tip_bus):
        features_list += bus_tip_type_avg[bus]
    elif (bus not in tip_bus) and (user in tip_user):
        features_list += user_tip_type_avg[user]
    else:
        features_list += tip_type_avg

    return features_list

# %%
def get_dict_median(rdd, bus_dict_len, photo = False, checkin = False):
    feature_dict = dict(rdd.collect())
    if photo:
        feature_avg = stat.median(rdd.map(lambda x: x[1][0]).collect() + [0] * (bus_dict_len - len(feature_dict)))
    elif checkin:
        feature_avg = [stat.median(rdd.map(lambda x: x[1][i]).collect() + [0] * (bus_dict_len - len(feature_dict))) for i in range(2)]
    else:
        feature_avg = stat.median(rdd.map(lambda x: x[1]).collect() + [0] * (bus_dict_len - len(feature_dict)))
    return feature_dict, feature_avg

# %%
def model_features(train_rdd, val_rdd, user_data, bus_data, photo_data, checkin_data, tip_data):
    user_data_dict = dict(user_data.map(lambda x: (x['user_id'], (x['average_stars'], x['review_count'], x['useful'], x['fans'], x['funny'], x['cool'], datetime.strptime(x['yelping_since'], '%Y-%m-%d').timestamp(), len(x['elite']), x['compliment_hot'], x['compliment_more'], x['compliment_cute'], x['compliment_note'], x['compliment_writer'], x['compliment_photos']))).collect())
    user_friends_dict = dict(user_data.filter(lambda x: x['friends'] != 'None').map(lambda x: (x['user_id'], len(x['friends']))).collect())
    
    bus_data_dict = dict(bus_data.map(lambda x: (x['business_id'], (x['stars'], x['review_count'], x['latitude'], x['longitude'], len(x.get('categories', '').split(',')) if x.get('categories') else 0))).collect())
    bus_attribute = bus_data.map(lambda x: (x['business_id'], x['attributes'])).collect()
    bus_values = list(bus_data_dict.values())
    bus_data_avg = [stat.mean([v for v in val if v is not None]) if i < len(bus_values) - 1 else stat.median([v for v in val if v is not None]) for i, val in enumerate(zip(*bus_values))]

    for (id, attr) in bus_attribute:
        if attr is not None:
            if 'DriveThru' in attr:
                bus_data_dict[id] += (1 if attr['DriveThru'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'RestaurantsPriceRange2' in attr: 
                bus_data_dict[id] += (int(attr['RestaurantsPriceRange2']),)            
            else:
                bus_data_dict[id] += (None,)
            if 'GoodForKids' in attr:
                bus_data_dict[id] += (1 if attr['GoodForKids'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'Caters' in attr:
                bus_data_dict[id] += (1 if attr['Caters'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'HasTV' in attr:
                bus_data_dict[id] += (1 if attr['HasTV'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'RestaurantsDelivery' in attr:
                bus_data_dict[id] += (1 if attr['RestaurantsDelivery'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'OutdoorSeating' in attr:
                bus_data_dict[id] += (1 if attr['OutdoorSeating'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'RestaurantsReservations' in attr:
                bus_data_dict[id] += (1 if attr['RestaurantsReservations'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'RestaurantsTableService' in attr:
                bus_data_dict[id] += (1 if attr['RestaurantsTableService'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'NoiseLevel' in attr:
                bus_data_dict[id] += (attr['NoiseLevel'],)
            else:
                bus_data_dict[id] += (None,)
            if 'WiFi'  in attr:
                bus_data_dict[id] += (attr['WiFi'],)
            else:
                bus_data_dict[id] += (None,)
            if 'DogsAllowed' in attr:
                bus_data_dict[id] += (1 if attr['DogsAllowed'] == 'True' else 0,)
            else:
                bus_data_dict[id] += (None,)
            if 'Smoking' in attr and attr['Smoking'] == 'yes':
                bus_data_dict[id] += (1,)
            elif 'Smoking' in attr:
                bus_data_dict[id] += (0,)
            else:
                bus_data_dict[id] += (None,)
        else:
            bus_data_dict[id] += (None,) * 13
    
    noise_dict = {None: None, 'very_loud': 0, 'loud': 1, 'average': 2, 'quiet': 3, 'very_quiet': 4}
    wifi_dict = {None: None, 'no': 0, 'free': 1, 'paid': 2}

    bus_dict_len = len(bus_data_dict)
    bus_photo_num = photo_data.groupBy(lambda x: x['business_id']).mapValues(lambda x: (len(x), [i['caption'] for i in x]))
    bus_photo_num_dict, bus_photo_num_avg = get_dict_median(bus_photo_num, bus_dict_len, photo = True)
    bus_photo_cap_dict = {key: len([cap for cap in val[1] if cap != ''])for key, val in bus_photo_num_dict.items()}
    bus_photo_cap_avg = stat.median([val for val in bus_photo_cap_dict.values()])

    checkin_sum = checkin_data.map(lambda x: (x['business_id'], (sum(x['time'].values()), len(x['time']))))
    checkin_sum_dict, checkin_sum_avg = get_dict_median(checkin_sum, bus_dict_len, checkin = True)
        
    tip_dict = dict(tip_data.map(lambda x: ((x['user_id'], x['business_id']), [len(x['text'].split()), datetime.strptime(x['date'], '%Y-%m-%d').timestamp()])).collect())
    tip_user = map(lambda k: k[0], tip_dict.keys())
    tip_bus = map(lambda k: k[1], tip_dict.keys())
    tip_dict_rdd = sc.parallelize(tip_dict.items())
    tip_type_avg = [tip_dict_rdd.map(lambda x: x[1][0]).mean(), tip_dict_rdd.map(lambda x: x[1][1]).mean()]
    user_tip_type_avg = dict(tip_dict_rdd.groupBy(lambda x: x[0][0]).mapValues(lambda x: [stat.mean([i[1][0] for i in x]), stat.mean([i[1][1] for i in x])]).collect())
    bus_tip_type_avg = dict(tip_dict_rdd.groupBy(lambda x: x[0][1]).mapValues(lambda x: [stat.mean([i[1][0] for i in x]), stat.mean([i[1][1] for i in x])]).collect())
    user_tip_num_dict = dict(tip_dict_rdd.groupBy(lambda x: x[0][0]).mapValues(lambda x: len(list(x))).collect())
    
    user_data_dict = {key: user_data_dict[key] + (user_friends_dict[key] if key in user_friends_dict else 0,) + (user_tip_num_dict[key] if key in user_tip_num_dict else 0,) for key in user_data_dict.keys()}
    user_data_avg = [stat.mean(val) for val in zip(*user_data_dict.values())]

    bus_tip_num = tip_dict_rdd.groupBy(lambda x: x[0][1]).mapValues(lambda x: len(list(x)))
    bus_tip_num_dict, bus_tip_num_avg = get_dict_median(bus_tip_num, bus_dict_len)

    bus_data_avg += [bus_photo_num_avg] + checkin_sum_avg + [bus_tip_num_avg, bus_photo_cap_avg]
    bus_data_dict = {key: val[:5] 
                        + (int(bus_photo_num_dict[key][0]) if key in bus_photo_num_dict else 0,) 
                        + (int(checkin_sum_dict[key][0]) if key in checkin_sum_dict else 0,) 
                        + (int(checkin_sum_dict[key][1]) if key in checkin_sum_dict else 0,) 
                        + (int(bus_tip_num_dict[key]) if key in bus_tip_num_dict else 0,) 
                        + (bus_photo_cap_dict[key] if key in bus_photo_cap_dict else 0,)
                        + val[5:14] + (noise_dict[val[14]],) + (wifi_dict[val[15]],) + val[16:] for key, val in bus_data_dict.items()}

    train_feature = train_rdd.map(lambda x: create_feature(x[0], x[1], user_data_dict, bus_data_dict, user_data_avg, bus_data_avg, tip_dict, user_tip_type_avg, bus_tip_type_avg, tip_type_avg, tip_user, tip_bus)).collect()
    val_feature = val_rdd.map(lambda x: create_feature(x[0], x[1], user_data_dict, bus_data_dict, user_data_avg, bus_data_avg, tip_dict, user_tip_type_avg, bus_tip_type_avg, tip_type_avg, tip_user, tip_bus)).collect()

    X_train = np.array(train_feature)
    y_train = np.array(train_rdd.map(lambda x: float(x[2])).collect())
    X_test = np.array(val_feature)

    return X_train, y_train, X_test

# %%
def model_based(X_train, y_train, X_test):

    model = XGBRegressor(objective = 'reg:linear', n_estimators = 1100, learning_rate = 0.1, random_state = 1, gamma = 0.1, min_child_weight = 2, max_depth = 4, alpha = 0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_pred

# %%
start_time = time.time()

user_json = sc.textFile(folder_path + 'user.json')
user_data = user_json.map(lambda x: json.loads(x))

bus_json = sc.textFile(folder_path + 'business.json')
bus_data = bus_json.map(lambda x: json.loads(x))

photo_json = sc.textFile(folder_path + 'photo.json')
photo_data = photo_json.map(lambda x: json.loads(x))

checkin_json = sc.textFile(folder_path + 'checkin.json')
checkin_data = checkin_json.map(lambda x: json.loads(x))

tip_json = sc.textFile(folder_path + 'tip.json')
tip_data = tip_json.map(lambda x: json.loads(x))

train = sc.textFile(folder_path + 'yelp_train.csv')
train_rdd = train.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0]).map(lambda x: x.split(','))

val = sc.textFile(test_file_name)
val_rdd = val.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0]).map(lambda x: x.split(','))
val_list = val_rdd.collect()

X_train, y_train, X_test = model_features(train_rdd, val_rdd, user_data, bus_data, photo_data, checkin_data, tip_data)
model_based_score = model_based(X_train, y_train, X_test)
item_based_score = item_based_cf(train_rdd, val_rdd)

ranged_item_based_s = [(x[0], x[1], 5 if x[2] > 5 else (1 if x[2] < 1 else x[2])) for x in item_based_score]
ranged_model_based_s = np.clip(model_based_score, 1, 5)
two_score_list = [i + ((j,)) for i, j in zip(ranged_item_based_s, ranged_model_based_s)]
output = []
for user, bus, item_s, model_s in two_score_list:
    new_score = 0.06781046 * item_s + 0.94000548 * model_s - 0.03073931836332333
    output.append((user, bus, new_score))
final_output = [(x[0], x[1], 5 if x[2] > 5 else (1 if x[2] < 1 else x[2])) for x in output]

with open(output_file_name, 'w') as file:
    file.write('user_id, business_id, prediction\n')
    for item in final_output:
        file.write(f"{item[0]},{item[1]},{item[2]}\n")

end_time = time.time()
print(f'Duration: {end_time - start_time}')