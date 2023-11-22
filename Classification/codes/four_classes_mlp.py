import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 定义一个class-wise的损失
class CrossEntropyLossWithWeights(nn.Module):
    def __init__(self, weight):
        super(CrossEntropyLossWithWeights, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        loss = F.cross_entropy(input, target, weight=self.weight)
        return loss

# ============================= TRAIN ===============================
# ================== DATA PREPARATION ====================
# 创建一个MinMaxScaler对象
scaler = MinMaxScaler()

# 读取表格数据
data_df = pd.read_csv('input_output_2cls.csv')
train_df, val_df = train_test_split(data_df, test_size=0.1, random_state=42)
test_df = pd.read_csv('test_2cls.csv')
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
features = train_df.iloc[:, [7,8,10,11]+list(range(12, train_df.shape[1]))].values
# features = train_df.iloc[:, [7,10]+list(range(12, train_df.shape[1]))].values
# features = train_df.iloc[:, 12:].values
# print(features)
val_file_names = val_df['name'].values
val_labels = val_df['targets'].values
val_features = val_df.iloc[:, [7,8,10,11]+list(range(12, val_df.shape[1]))].values
# val_features = val_df.iloc[:, [7,10]+list(range(12, val_df.shape[1]))].values
# val_features = val_df.iloc[:, 12:].values

test_file_names = test_df['name'].values
test_labels = test_df['targets'].values
test_features = test_df.iloc[:, [7,8,10,11]+list(range(12, test_df.shape[1]))].values
# test_features = test_df.iloc[:, [7,10]+list(range(12, test_df.shape[1]))].values
# test_features = test_df.iloc[:, 12:].values
# 统计train_data中每个类别的样本数量
class_counts = train_df['targets'].value_counts()
print(class_counts)
print(class_counts[0])

labels = torch.tensor(labels, dtype=torch.long)
features = torch.tensor(features, dtype=torch.float)
val_labels = torch.tensor(val_labels, dtype=torch.long)
val_features = torch.tensor(val_features, dtype=torch.float)
test_labels = torch.tensor(test_labels, dtype=torch.long)
test_features = torch.tensor(test_features, dtype=torch.float)

# 创建一个TensorDataset对象
dataset = TensorDataset(features, labels)
val_dataset = TensorDataset(val_features, val_labels)
test_dataset = TensorDataset(test_features, test_labels)
# 创建一个DataLoader对象
batch_size = 128
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



input_size = features.shape[1]
hidden_size = 20
output_size = 1  # 四分类任务
mlp = MLP(input_size, hidden_size, output_size)

# 定义损失函数和优化器
# mlp_criterion = nn.CrossEntropyLoss() # 四分类
mlp_criterion = nn.BCEWithLogitsLoss() # 二分类
# 按样本数量设置每个类别的代价权重
# class_weights = torch.tensor([1/class_counts[0], 1/class_counts[1], 1/class_counts[2], 1/class_counts[3]]).float()
# class_weights = torch.tensor([1.0, 0.5, 1.0, 4.0])
# mlp_criterion = CrossEntropyLossWithWeights(class_weights)


mlp_optimizer = optim.Adam(mlp.parameters(), lr=1e-4)

# ============================= TEST ===============================
def test():
    # 在验证集上评估模型
    mlp.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = mlp(inputs)
            # _, predicted = torch.max(outputs, 1) # 四分类
            predicted = (outputs > 0).view(-1)  # 二分类
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 将当前批次的预测和标签添加到列表中
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy}%")
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1 Score: {f1}")

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    # report = classification_report(all_labels, all_predictions, target_names=["II", "III", "IV", "V"])
    report = classification_report(all_labels, all_predictions, target_names=["0", "1"])

    return accuracy, report


num_epochs = 200
losses = []
test_interval=10
accuracies=[]
best_accuracy = 0.0
for epoch in range(num_epochs):
    mlp.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        mlp_optimizer.zero_grad()  # 清零梯度

        outputs = mlp(inputs)
        # outputs = torch.tensor(outputs,dtype=torch.float)
        # print(outputs.dtype)
        # print(labels.dtype)

        # loss = mlp_criterion(outputs, labels) # 四分类
        loss = mlp_criterion(outputs, labels.float().view(-1, 1))  # 二分类
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        mlp_optimizer.step()
        running_loss += loss.item()

    # 计算平均损失并记录
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    if epoch%test_interval==0:
        acc, report = test()
        accuracies.append(acc)

        if acc > best_accuracy:
            # 如果是，保存模型权重
            best_accuracy = acc
            torch.save(mlp.state_dict(), "best_mlp_model.pth")

best_mlp = MLP(input_size, hidden_size, output_size)  # 请替换为你的MLP模型定义
best_mlp.load_state_dict(torch.load("best_mlp_model.pth"))
best_mlp.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = best_mlp(inputs)
        # _, predicted = torch.max(outputs, 1)
        predicted = (outputs > 0).view(-1)  # 二分类
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 将当前批次的预测和标签添加到列表中
        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy}%")

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
# report = classification_report(all_labels, all_predictions, target_names=["II", "III", "IV", "V"])
report = classification_report(all_labels, all_predictions, target_names=["0", "1"])
print(report)

conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

f1 = f1_score(all_labels, all_predictions, average='weighted')
print(f"F1 Score: {f1}")

# 绘制训练损失曲线
plt.figure(figsize=(8, 6))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Loss/loss_64mlp_500e_128bs_-4lr_test.png')
plt.close()


# # 绘制训练损失曲线
# plt.figure(figsize=(8, 6))
# plt.plot(accuracies)
# plt.title('Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Precision')
# plt.savefig('Acc/acc_64mlp_500e_128bs_-4lr.png')
# plt.close()