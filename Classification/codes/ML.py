import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

# 创建一个MinMaxScaler对象
scaler = MinMaxScaler()

# 读取表格数据
data_df = pd.read_csv('input_output.csv')
train_df, val_df = train_test_split(data_df, test_size=0.1, random_state=42)
test_df = pd.read_csv('test.csv')
merged_df = pd.concat([data_df , test_df], axis=1)
# 从第三列开始正则化每一列数据
start_column_index = 2  # 第三列的索引为2
start_row_index = 1  # 第三列的索引为2
# 选择需要正则化的列范围，从第三列到最后一列
train_columns_to_normalize = train_df.columns[start_column_index:]
val_columns_to_normalize = val_df.columns[start_column_index:]
test_columns_to_normalize = test_df.columns[start_column_index:]

scaler.fit(train_df[train_columns_to_normalize])

train_df[train_columns_to_normalize] = scaler.transform(train_df[train_columns_to_normalize])
val_df[val_columns_to_normalize] = scaler.transform(val_df[val_columns_to_normalize])
test_df[test_columns_to_normalize] = scaler.transform(test_df[test_columns_to_normalize])

# 提取文件名、标签和特征数据
file_names = train_df['name'].values
labels = train_df['targets'].values
# features = train_df.iloc[:, [13, 14, 17,18,27,28]].values
# features = train_df.iloc[:, [7,8,10,11]+list(range(12, train_df.shape[1]))].values
features = train_df.iloc[:, [7,10]+list(range(12, train_df.shape[1]))].values
# features = train_df.iloc[:, 12:].values
# print(features)
val_file_names = val_df['name'].values
val_labels = val_df['targets'].values
# val_features = val_df.iloc[:, [13, 14, 17,18,27,28]].values
# val_features = val_df.iloc[:, [7,8,10,11]+list(range(12, val_df.shape[1]))].values
val_features = val_df.iloc[:, [7,10]+list(range(12, val_df.shape[1]))].values
# val_features = val_df.iloc[:, 12:].values

test_file_names = test_df['name'].values
test_labels = test_df['targets'].values
# test_features = test_df.iloc[:, [13, 14, 17,18,27,28]].values
# test_features = test_df.iloc[:, [7,8,10,11]+list(range(12, test_df.shape[1]))].values
test_features = test_df.iloc[:, [7,10]+list(range(12, test_df.shape[1]))].values
# test_features = test_df.iloc[:, 12:].values
# 统计train_data中每个类别的样本数量
class_counts = train_df['targets'].value_counts()
print(class_counts)
print(class_counts[0])


from sklearn.cluster import KMeans

num_clusters = 2  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(features.numpy())  # You may need to convert the features to NumPy

from sklearn.svm import SVC

# Assuming you have your labels stored in 'train_labels'
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(features.numpy(), labels.numpy())


from sklearn.metrics import accuracy_score
from collections import Counter

kmeans_pred = kmeans.predict(val_features.numpy())  # Predict using k-means
svm_pred = svm_classifier.predict(val_features.numpy())  # Predict using SVM

# 假设kmeans_pred包含K-Means的簇分配结果
# 假设true_labels包含真实的类别标签
cluster_to_label = {}

for cluster in set(kmeans_pred):
    cluster_indices = [i for i, c in enumerate(kmeans_pred) if c == cluster]
    cluster_labels = [val_labels[i] for i in cluster_indices]
    most_common_label = Counter(cluster_labels).most_common(1)[0][0]
    cluster_to_label[cluster] = most_common_label

kmeans_pred_mapped = np.array([cluster_to_label[cluster] for cluster in kmeans_pred])

# print(kmeans_pred_mapped)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# Assuming you already have extracted features stored in 'autoencoder_features'
# Assuming you have labels for the training data stored in 'labels'

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(features, labels)

# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=0)
dt_classifier.fit(features, labels)

# Now, validate the models using val_loader
rf_predictions = rf_classifier.predict(val_features)
dt_predictions = dt_classifier.predict(val_features)

from sklearn.ensemble import GradientBoostingClassifier

# Initialize and train the Gradient Boosting Decision Tree (GBDT) Classifier
gbdt_classifier = GradientBoostingClassifier(n_estimators=100, random_state=0)
gbdt_classifier.fit(features, labels)

# Now, validate the GBDT model using val_loader
gbdt_predictions = gbdt_classifier.predict(val_features)


# 4-cls classification
# knn_report = classification_report(val_labels, kmeans_pred_mapped, target_names=["II", "III", "IV", "V"])
# svm_report = classification_report(val_labels, svm_pred, target_names=["II", "III", "IV", "V"])
# rf_report = classification_report(val_labels, rf_predictions, target_names=["II", "III", "IV", "V"])
# dt_report = classification_report(val_labels, dt_predictions, target_names=["II", "III", "IV", "V"])
# gbdt_report = classification_report(val_labels, gbdt_predictions, target_names=["II", "III", "IV", "V"])

# 2-cls classification
knn_report = classification_report(val_labels, kmeans_pred_mapped, target_names=["0", "1"])
svm_report = classification_report(val_labels, svm_pred, target_names=["0", "1"])
rf_report = classification_report(val_labels, rf_predictions, target_names=["0", "1"])
dt_report = classification_report(val_labels, dt_predictions, target_names=["0", "1"])
gbdt_report = classification_report(val_labels, gbdt_predictions, target_names=["0", "1"])

# Accuracy
kmeans_accuracy = accuracy_score(val_labels, kmeans_pred_mapped)
svm_accuracy = accuracy_score(val_labels, svm_pred)
rf_accuracy = accuracy_score(val_labels, rf_predictions)
dt_accuracy = accuracy_score(val_labels, dt_predictions)
gbdt_accuracy = accuracy_score(val_labels, gbdt_predictions)

print(f'K-Means Accuracy: {kmeans_accuracy}, K-Means Accuracy: {knn_report}')
print(f'SVM Accuracy: {svm_accuracy}, SVM Accuracy: {svm_report}')
print(f'Random Forest Accuracy: {rf_accuracy}, Random Forest Accuracy: {rf_report}')
print(f'Decision Tree Accuracy: {dt_accuracy}, Decision Tree Accuracy: {dt_report}')
print(f'GBDT Accuracy: {gbdt_accuracy}, GBDT Accuracy: {gbdt_report}')

#############################################
# ===========4-cls classification============
#############################################

# =============== with resistance ===================
# [7,10]
# K-Means Accuracy: 0.4608
# SVM Accuracy: 0.7168
# Random Forest Accuracy: 0.8
# Decision Tree Accuracy: 0.5184
# GBDT Accuracy: 0.7184

# [7,8,10,11]
# K-Means Accuracy: 0.504
# SVM Accuracy: 0.7136
# Random Forest Accuracy: 0.776
# Decision Tree Accuracy: 0.5728
# GBDT Accuracy: 0.72

# =============== without resistance ===================
# K-Means Accuracy: 0.4112
# SVM Accuracy: 0.7136
# Random Forest Accuracy: 0.7872
# Decision Tree Accuracy: 0.6048
# GBDT Accuracy: 0.7264


#############################################
# ===========2-cls classification============
#############################################

# =============== with proof ===================
# [7,8,10,11]
# K-Means Accuracy: 0.2784
# SVM Accuracy: 0.9024
# Random Forest Accuracy: 0.9184
# Decision Tree Accuracy: 0.8176
# GBDT Accuracy: 0.9072

# [7,10]
# K-Means Accuracy: 0.2784
# SVM Accuracy: 0.8992
# Random Forest Accuracy: 0.9104
# Decision Tree Accuracy: 0.8496
# GBDT Accuracy: 0.8912

# =============== without proof ===================
# K-Means Accuracy: 0.2784
# SVM Accuracy: 0.896
# Random Forest Accuracy: 0.9136
# Decision Tree Accuracy: 0.864
# GBDT Accuracy: 0.8944





