import os
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
import numpy as np
import pandas as pd


import torch

def MRE(labels, pred):
    # 计算绝对误差
    absolute_errors = np.abs(labels - pred)

    # 计算相对误差（MRE）
    relative_errors = absolute_errors / labels

    # 计算平均相对误差
    mean_relative_error = np.mean(relative_errors)

    return mean_relative_error

def r2_score(labels, pred):
    # 计算绝对误差
    error_square = (labels - pred)**2

    labels_square = (labels-np.mean(labels))**2

    r2 = 1- np.sum(error_square)/np.sum(labels_square)

    return r2

folder_path = 'regression/train/F_train'

file_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

# 创建一个空的DF来存储所有样本
all_samples = pd.DataFrame(columns=['刀盘转速', '推进速度(nn/M)', '刀盘扭矩', '总推力'])

for file in file_list:

    df = pd.read_csv(file)
    # print(df)

    # 提取特征
    features = df[['刀盘转速', '推进速度(nn/M)', '刀盘扭矩']]
    target = df['总推力']

    # 将每个表格的数据添加到总体样本集
    sample = pd.DataFrame({'刀盘转速': features['刀盘转速'],
                            '推进速度(nn/M)': features['推进速度(nn/M)'],
                           '刀盘扭矩': features['刀盘扭矩'],
                            '总推力': target})
    all_samples = pd.concat([all_samples, sample], ignore_index=True)

# 数据预处理
print(all_samples) # 目标
all_samples['刀盘扭矩'] = all_samples['刀盘扭矩'] / 34
all_samples['总推力'] = all_samples['总推力'] / 34
all_samples = all_samples[all_samples['刀盘扭矩'] > 0]
all_samples = all_samples[all_samples['总推力'] > 0]

print(all_samples)
target_cols=["总推力"]
# target_cols = ['刀盘扭矩']
# train, test = train_test_split(all_samples, random_state=42)
train = all_samples
X_train = train[['刀盘转速', '推进速度(nn/M)', '总推力']]
# y_train = train['总推力']
# X_test = test[['刀盘转速', '推进速度(nn/M)', '刀盘扭矩']]
# y_test = test['总推力']
batch_size = 128
steps_per_epoch = int(train.shape[0]/batch_size)

# target_col="targets"
cat_col_names = [] # Assuming no categorical features in this case
num_col_names = ['刀盘转速', '推进速度(nn/M)', '刀盘扭矩']
# num_col_names = ['刀盘转速', '推进速度(nn/M)', '总推力']

data_config = DataConfig(
    target=target_cols,
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)

optimizer_config = OptimizerConfig(lr_scheduler="OneCycleLR", lr_scheduler_params={"max_lr":0.01, "epochs": 20, "steps_per_epoch":steps_per_epoch})

head_config = LinearHeadConfig(
    layers="",
    dropout=0.1,
    initialization="kaiming"
).__dict__

model_config = CategoryEmbeddingModelConfig(
    task="regression",
    layers="64-32-16",
    activation="LeakyReLU",
    dropout=0.1,
    initialization="kaiming",
    head = "LinearHead",
    head_config = head_config,
    learning_rate = 1e-3,
    target_range=[(float(train[col].min()),float(train[col].max())) for col in target_cols] # 总推力
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config="./trainer_cfg.yml",
)

# Check if multiple GPUs are available and if so, use DataParallel
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    tabular_model = torch.nn.DataParallel(tabular_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datamodule = tabular_model.prepare_dataloader(
                train=train, validation=train, seed=42
            )
model = tabular_model.prepare_model(
            datamodule
        )
tabular_model.train(model, datamodule)


# TEST
test_folder_path = 'regression/test/F'
test_file_list = [os.path.join(test_folder_path, file) for file in os.listdir(test_folder_path) if os.path.isfile(os.path.join(test_folder_path, file))]

for file in test_file_list:

    print(file)
    test_df = pd.read_excel(file)

    test = test_df[['刀盘转速', '总推力', '推进速度(nn/M)', '刀盘扭矩']]

    # test['总推力'] = test['总推力'] / 34
    test['刀盘扭矩'] = test['刀盘扭矩'] / 34

    result = tabular_model.evaluate(test)

    pred_df = tabular_model.predict(test)
    # pred_df.head()

    # print("刀盘扭矩")
    # mre = MRE(test_df['刀盘扭矩'], pred_df["刀盘扭矩_prediction"])
    # print(f"Holdout MRE: {mre}")
    # r2 = r2_score(test_df['刀盘扭矩'], pred_df["刀盘扭矩_prediction"])
    # print(f"Holdout R2_SCORE: {r2}")

    print("总推力")
    mre = MRE(test_df['总推力'], pred_df["总推力_prediction"])
    print(f"Holdout MRE: {mre}")
    r2 = r2_score(test_df['总推力'], pred_df["总推力_prediction"])
    print(f"Holdout R2_SCORE: {r2}")

    copied_test_df = test_df.copy()

    # 替换复制表格中的数据
    # 扭矩
    # output_list = pred_df["刀盘扭矩_prediction"]
    # copied_test_df['单刀扭矩预测值(kNm)'] = output_list
    # 推力
    output_list = pred_df["总推力_prediction"]
    copied_test_df['单刀推力预测值(kN)'] = output_list
    # 优度和误差
    copied_test_df['拟合优度'][1] = r2
    copied_test_df['相对误差'][1] = mre

    # 保存复制后的表格为execl文件
    output_filename = f"{file[:-5]}_prediction.xlsx"
    copied_test_df.to_excel(output_filename, index=False)



