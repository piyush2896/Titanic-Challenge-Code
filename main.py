from preprocessor import pre_process
import numpy as np
import pandas as pd

"""General Pre Processing"""
label = "Survived"
remove_list = ["Name", 
                "Ticket"]
di_male_female = {
    'male':1, 
    'female':2
}
di_embark = {
    'C':1, 
    'Q':2, 
    'S':3
}

from_to_values_list = [di_male_female, di_embark]

features_train, labels_train = pre_process("train.csv", remove_list, from_to_values_list, label_name=label, 
                                remove_nan_points=False, convert_nan_zero=True)

features_test = pre_process("test.csv", remove_list, from_to_values_list, remove_nan_points=False, 
                    convert_nan_zero=True)


# Calculate total family members and add to features list
features_train = np.c_[features_train[:, [0, 1, 2, 3, 6, 7]], features_train[:, 4] + features_train[:, 5]]
features_test = np.c_[features_test[:, [0, 1, 2, 3, 6, 7]], features_test[:, 4] + features_test[:, 5]]

"""Sanity Check"""
print(features_train[10])
print(features_test[10])

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train[:, 1:], labels_train)

"""Sanity Check"""
from sklearn.metrics import accuracy_score
print(accuracy_score(labels_train, clf.predict(features_train[:, 1:])))

"""Predictions"""
preds = clf.predict(features_test[:, 1:])

"""Save Files to result.csv"""
result_dict = {
    "PassengerId" : features_test[:, [0]].ravel().astype(int),
    "Survived" : preds.astype(int)
}

result_df = pd.DataFrame(result_dict)
result_df.to_csv("result.csv", index=False)