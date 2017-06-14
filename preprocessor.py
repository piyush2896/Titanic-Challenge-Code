import pandas as pd
import numpy as np

def extract_data(file, remove_list, label_name=None, nan_threshold=0.6):
    dataframe = pd.DataFrame.from_csv(file, index_col=None)
    keys, removed_keys = remove_nan_features(dataframe, label_name, nan_threshold)
    for ele in remove_list:
        if ele not in removed_keys:
            keys.remove(ele)
    # to know all the column names uncomment next line
    print(keys)
    if label_name != None:
        labels_train = np.array(dataframe[[label_name]])
    
    features_train = []
    for index, row in dataframe.iterrows():
        features_train.append([row[key] for key in keys])
    features_train = np.array(features_train)
    if label_name == None:
        return features_train
    return features_train, labels_train

def remove_nan_features(dataframe, label_name, nan_threshold = 0.6):
    keys = list(dataframe)
    if label_name != None:
        keys.remove(label_name)
    new_keys = []
    removed_keys = []
    for key in keys:
        total_nan = 0
        arr = np.array(dataframe[[key]]).ravel()
        for i in range(len(arr)):
            if str(arr[i]) == "nan":
                total_nan += 1
        fraction = total_nan / len(arr)
        if fraction < nan_threshold:
            new_keys.append(key)
        else:
            removed_keys.append(keys)
    return new_keys, removed_keys

def remove_nan_data_points(features_train, labels_train=[]):
    i = 0
    features_train = features_train.astype(str)
    if len(labels_train) != 0:
        labels_train = labels_train.astype(str)
    while i < len(features_train):
        if "nan" in features_train[i]:
            features_train = np.delete(features_train, (i), axis=0)
            if len(labels_train) != 0:
                labels_train = np.delete(labels_train, (i), axis=0)
            i -= 1
        i += 1
    if len(labels_train) == 0:
        return features_train
    return features_train, labels_train

def convert_vals(features_train, from_to_values):
    new_features_train = np.copy(features_train)
    for key, val in from_to_values.items():
        new_features_train[features_train == key] = val
    return new_features_train

def convert_nan_to_zeros(data):
    new_data = np.copy(data)
    new_data[data=="nan"] = 0
    return new_data

def pre_process(file, remove_list, from_to_values_list, label_name = None, 
                nan_threshold=0.6, remove_nan_points=True, convert_nan_zero=False):
    if label_name != None:
        features_train, labels_train = extract_data(file, remove_list, label_name, nan_threshold)
    else:
        features_train= extract_data(file, remove_list, label_name, nan_threshold)
    
    if remove_nan_points:
        if label_name != None:
            features_train, labels_train = remove_nan_data_points(features_train, labels_train)
        else:
            features_train = remove_nan_data_points(features_train)
    
    if convert_nan_zero:
        features_train = convert_nan_to_zeros(features_train)
    
    for item in from_to_values_list:
        features_train = convert_vals(features_train, item)
    
    if label_name != None:
        return features_train.astype(float), labels_train.astype(float).ravel()
    else:
        return features_train.astype(float)

# Testing
if __name__ == '__main__':
    label = "Survived"
    remove_list = ["Name", "Ticket"]
    di_male_female = {'male':1, 'female':2}
    di_embark = {'C':1, 'Q':2, 'S':3}
    from_to_values_list = [di_male_female, di_embark]
    features_train, labels_train = pre_process("train.csv", remove_list, from_to_values_list, label_name=label, 
                                    remove_nan_points=False, convert_nan_zero=True)
    features_test = pre_process("test.csv", remove_list, from_to_values_list, remove_nan_points=False, 
                        convert_nan_zero=True)
    
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf')
    clf.fit(features_train, labels_train)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(labels_train, clf.predict(features_train)))