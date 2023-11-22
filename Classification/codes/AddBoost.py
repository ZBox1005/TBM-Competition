from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

from dataset import features, test_features, labels, test_labels

# 初始化数据，autoencoder_features 存储特征，labels 存储标签
# 将数据分为训练集和验证集

# 初始化分类器
svm_classifier = SVC(kernel='linear')
kmeans = KMeans(n_clusters=4, random_state=0)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
dt_classifier = DecisionTreeClassifier(max_depth=4, random_state=0)  # 弱决策树
gbdt_classifier = GradientBoostingClassifier(n_estimators=100, random_state=0)
lr_classifier = LogisticRegression(solver='lbfgs', multi_class='multinomial')

# 自定义 AdaBoostClassifier 类
class MyAdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=4, random_state=None):
        self.n_estimators = n_estimators
        self.estimators = []  # 存储估算器
        self.estimator_weights = []  # 存储估算器权重
        self.estimator_errors = []  # 存储估算器错误
        self.classes_ = None
        self.n_classes_ = None
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        sample_weights = np.ones(len(y)) if sample_weight is None else sample_weight

        # 训练 AdaBoost 分类器
        classifiers = [dt_classifier, gbdt_classifier]

        # 初始化分类器权重，初始时都设置为1
        classifier_weights = np.ones(len(classifiers))
        for i, clf in enumerate(classifiers):

            if clf==kmeans:
                clf.fit(X)
                predictions = clf.predict(X)

                cluster_to_label = {}

                for cluster in set(predictions):
                    cluster_indices = [i for i, c in enumerate(predictions) if c == cluster]
                    cluster_labels = [labels[i] for i in cluster_indices]
                    most_common_label = Counter(cluster_labels).most_common(1)[0][0]
                    cluster_to_label[cluster] = most_common_label

                predictions = np.array([cluster_to_label[cluster] for cluster in predictions])
            else:
                clf.fit(X, y, sample_weight=sample_weights)

                predictions = clf.predict(X)

            errors = (predictions != y).astype(int)
            print(np.sum(errors))
                
            error_rate = np.sum(errors * sample_weights) / np.sum(sample_weights)

            if clf == rf_classifier and error_rate == 0:
                # 当处理 Random Forest 分类器时
                # 如果错误率为0，将该分类器的权重设置为一个小的非零值，以避免过分强调它
                clf_weight = 0.1
            else:
                clf_weight = np.log((1 - error_rate) / error_rate)

            print(clf, error_rate, clf_weight)

            sample_weights *= np.exp(clf_weight * errors)
            sample_weights /= np.sum(sample_weights)
            sample_weights = np.nan_to_num(sample_weights, nan=1.0)

            self.estimators.append(clf)
            self.estimator_weights.append(clf_weight)
            self.estimator_errors.append(error_rate)

            # print(clf.classes_)

            if i == 0:
                self.classes_ = [0,1,2,3]
                self.n_classes_ = len(self.classes_)

        return self

    def predict(self, X):
        if not self.n_classes_:
            raise ValueError("Estimator not fitted, call `fit` before making predictions")
        final_predictions = np.zeros((X.shape[0], self.n_classes_))
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions = estimator.predict(X)
            for i, label in enumerate(self.classes_):
                final_predictions[:, i] += weight * (predictions == label)
        return np.argmax(final_predictions, axis=1)

# 初始化 AdaBoost 分类器
adaboost_classifier = MyAdaBoostClassifier(base_estimator=None, n_estimators=4, random_state=0)

# 训练 AdaBoost 分类器
features = np.array(features)  # 将特征转换为NumPy数组
labels = np.array(labels)  # 将标签转换为NumPy数组
sample_weights = np.ones(len(labels))  # 初始化样本权重

# 训练分类器
adaboost_classifier.fit(features, labels, sample_weight=sample_weights)

# 预测并评估结果
adaboost_predictions = adaboost_classifier.predict(test_features)
adaboost_accuracy = accuracy_score(test_labels, adaboost_predictions)
report = classification_report(test_labels, adaboost_predictions, target_names=["II", "III", "IV", "V"])
print(f'AdaBoost 准确率: {adaboost_accuracy}')
print(report)




