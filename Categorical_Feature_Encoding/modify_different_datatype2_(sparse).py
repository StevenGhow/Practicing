# Oct 21, 2019
# Kaggle challenge from https://www.kaggle.com/c/cat-in-the-dat
# according to different datatype to deal with the dataset
# accuracy: 0.8011 in test


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.


catTrain = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
catTest = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")
cat = pd.concat([catTrain, catTest], axis = 0, ignore_index = True, sort = False)
print(cat.shape) # number of rows and columns. (rows, cols)
print(cat.tail(5))

# see distribution of target
count_target1 = len(catTrain[catTrain["target"] == 1])
count_target0 = len(catTrain[catTrain["target"] == 0])
total = catTrain.shape[0]
print(count_target0/total, count_target1/total)

# View data in each columns
'''
bin_0 ~ bin_4: binary data
nom_0: colors
nom_1: shapes
nom_2: animal types
nom_3: countries
nom_4: instruments
nom_5 ~ nom_9: strings
ord_0: 1, 2, 3
ord_1: Novice, Contributor, Master, Expert, Grandmaster(0-4)
ord_2: Freezing, Cold, Warm, Hot, Boiling Hot, Lava Hot(0-5)
ord_3: a-o (0-14)
ord_4: A-Z (0-25)
ord_5: double letters (e.g. "av", "PZ", "jS", "Ed")
day: 1-7
month: 1-12
'''

# change values according to data types
# for binary data (bin0 - bin4)
bin_col = [col for col in cat.columns if "bin" in col]

# one hot encoding to binary data
bin_data = pd.DataFrame()
for i in bin_col:
    temp = pd.get_dummies(cat[i], drop_first = True)
    bin_data = pd.concat([bin_data, temp], axis = 1)
bin_data.columns = bin_col
bin_data

# modify values in ord_0 - ord_4
# see alphebat as sequence data
# ord_0 does not need to be converted
# for ord_1: Novice, Contributor, Master, Expert, Grandmaster(0-4)
# according to mean of targets grouped by ord_1 to figure out degrees between the five levels
order1 = catTrain.groupby("ord_1").mean()["target"].sort_values()
order1[range(0, 5)] = range(0, 5)
ord1 = [order1[i] for i in cat["ord_1"]]

# for ord_2: Freezing, Cold, Warm, Hot, Boiling Hot, Lava Hot(0-5)
# according to mean of targets grouped by ord_1 to figure out degrees between the five levels
order2 = catTrain.groupby("ord_2").mean()["target"].sort_values()
order2[range(0, 6)] = range(0, 6)
ord2 = [order2[i] for i in cat["ord_2"]]
ord2

# for ord_3: change alphebat to numbers
order3 = catTrain.groupby("ord_3").mean()["target"].sort_values()
order3[range(0, 15)] = range(0, 15)
ord3 = [order3[i] for i in cat["ord_3"]]
ord3

# for ord_4: change alphebat to numbers
order4 = catTrain.groupby("ord_4").mean()["target"].sort_values()
order4[range(0, 26)] = range(0, 26)
ord4 = [order4[i] for i in cat["ord_4"]]
order4

# for ord_5
order5 = catTrain.groupby("ord_5").mean()["target"].sort_values()
ord5_deg = set(list(cat["ord_5"]))
order5[range(len(ord5_deg))] = range(len(ord5_deg))
ord5 = [order5[i] for i in cat["ord_5"]]
ord5

# combine ord_0 to ord_5
ord1, ord2, ord3, ord4, ord5 = pd.DataFrame(ord1), pd.DataFrame(ord2), pd.DataFrame(ord3), pd.DataFrame(ord4), pd.DataFrame(ord5)
ord_data = pd.concat([cat["ord_0"], ord1, ord2, ord3, ord4, ord5], axis = 1)
ord_data.columns = [col for col in cat.columns if "ord" in col]
ord_data

# periodic data: day and month
period_data = pd.concat([cat["day"], cat["month"]], axis = 1)
period_data

# modify values in nom_5 - nom_9
nom_col = [col for col in cat.columns if "nom" in col and col != "nom_9"]
nom_data = cat[nom_col]

# combine all kinds of data
n_cat = pd.concat([bin_data, ord_data, period_data, nom_data], axis = 1)
n_cat.head(10)
del [bin_data, ord_data, period_data, nom_data]

# one hot encoding except for nom_9 (because of too many candidates)
n_cat = pd.get_dummies(n_cat, columns = nom_col, drop_first = True)
del cat


# resampling
from scipy.sparse import vstack, csr_matrix
from imblearn.over_sampling import RandomOverSampler
x_train = csr_matrix(n_cat[:300000])
y_train = catTrain["target"][:300000]
test = csr_matrix(n_cat[catTrain.shape[0]:])
'''ros = RandomOverSampler(random_state = 0)
x_res, y_res = ros.fit_resample(x_train, y_train)'''

# from sklearn.feature_selection import RFE # Recursive Feature Elimination
from sklearn.linear_model import LogisticRegression
logModel = LogisticRegression()
logModel.fit(x_train, y_train)

# predict test data
predictions = logModel.predict_proba(test)
predict_df = pd.DataFrame(predictions)

predict_df.columns = ["0_prob", "target"]
out = pd.concat([catTest["id"], predict_df["target"]], axis = 1)
out.head(5)

import os
os.getcwd()
out.to_csv("Result_3.csv", index=0)