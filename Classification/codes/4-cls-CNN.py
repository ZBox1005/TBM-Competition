import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score


# 构建简单的一维CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.maxpool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(16 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 将数据形状从 (batch_size, 20) 调整为 (batch_size, 1, 20)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 16 * 7)  # 将卷积输出展平
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def validation(model):
    # 在验证集上评估模型
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = model(inputs)
            # _, predicted = torch.max(logits, 1) # 四分类
            predicted = (logits > 0).view(-1) # 二分类
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
    # print(report)

    return accuracy, report

# 训练和测试流程
def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, learning_rate):
    # criterion = nn.CrossEntropyLoss() # 四分类
    criterion = nn.BCEWithLogitsLoss() # 二分类
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_accuracy = 0.0
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            # loss = criterion(logits, labels)
            loss = criterion(logits, labels.float().view(-1, 1))
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

        # 在测试集上评估模型
        if (epoch + 1) % test_interval == 0:
            acc, report = validation(model)
            accuracies.append(acc)
            # 检查是否当前验证精度超过了最佳验证精度
            if acc > best_accuracy:
                # 如果是，保存模型权重
                best_accuracy = acc
                torch.save(model.state_dict(), "best_mlp_model.pth")


from dataset import train_loader, val_loader, test_loader, features
# 初始化模型
input_dim = features.shape[1] # 输入特征的维度
num_classes = 1  # 输出类别数
model = SimpleCNN(num_classes)
test_interval=10

# 训练和测试模型
train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs=200, learning_rate=1e-4)

last_model=model
last_model.eval()
best_model = SimpleCNN(num_classes)  # 请替换为你的MLP模型定义
best_model.load_state_dict(torch.load("best_mlp_model.pth"))
best_model.eval()
correct = 0
total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        logits = best_model(inputs)
        # _, predicted = torch.max(logits, 1) # 四分类
        predicted = (logits > 0).view(-1)  # 二分类
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 将当前批次的预测和标签添加到列表中
        all_predictions.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

# last_model
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         logits = last_model(inputs)
#         _, predicted = torch.max(logits, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#         # 将当前批次的预测和标签添加到列表中
#         all_predictions.extend(predicted.tolist())
#         all_labels.extend(labels.tolist())

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
