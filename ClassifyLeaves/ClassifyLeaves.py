import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# This is for the progress bar.
from tqdm import tqdm

train_path = './train.csv'
test_path = './test.csv'
img_path = './'


class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='Train', valid_ratio=0.2, length_resize=256, width_resize=256):
        self.length_resize = length_resize
        self.width_resize = width_resize
        self.img_path = file_path
        self.mode = mode

        # 读取csv文件
        self.all_data = pd.read_csv(csv_path, header=None)

        # 计算长度
        self.data_len = len(self.all_data.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'Train':
            self.img_arr = np.asarray(self.all_data.iloc[1:self.train_len, 0])
            self.labels_arr = np.asarray(self.all_data.iloc[1:self.train_len, 1])
        elif mode == 'Valid':
            self.img_arr = np.asarray(self.all_data.iloc[self.train_len:, 0])
            self.labels_arr = np.asarray(self.all_data.iloc[self.train_len:, 1])
        elif mode == 'Test':
            self.img_arr = np.asarray(self.all_data.iloc[1:, 0])

        self.real_len = len(self.img_arr)
        print(f'{mode} dataset has loaded, {self.real_len} samples found')

    def __getitem__(self, index):
        sample_name = self.img_arr[index]
        img_item = Image.open(self.img_path + sample_name)

        if self.mode == 'Train':
            # 给训练数据做数据增强
            transforms_mode = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(size=(224 * 0.8), scale=(0.5, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
                transforms.RandomRotation(degrees=360),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            transforms_mode = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        final_img = transforms_mode(img_item)

        if self.mode == 'Test':
            return final_img
        else:
            label = self.labels_arr[index]
            final_label = labels2num[label]
            return final_img, final_label

    def __len__(self):
        return self.real_len


# 读取数据
train_dataframe = pd.read_csv(train_path)
test_dataframe = pd.read_csv(test_path)

labels_list = sorted(list(set(train_dataframe['label'])))
n_labels = len(labels_list)
# 将标签转化为数字
labels2num = dict(zip(labels_list, range(n_labels)))
# 转化回来也需要
num2labels = {v: k for k, v in labels2num.items()}


def im_convert(tensor):
    """ 展示数据"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)

    return image


"""resnet34模型"""


def res_mosdels(num_classes, pre_train=True):
    res_model = models.resnet18(pretrained=pre_train)
    num_in = res_model.fc.in_features
    res_model.fc = nn.Sequential(nn.Linear(num_in, num_classes))

    return res_model


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


best_acc = 0.0
model_path = './models/resnet34.ckpt'


def train_model(net, train_iter, valid_iter, num_epochs, lr, device):
    global best_acc
    net.to(device)
    print(f'train on {device}')
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    for epoch in range(num_epochs):
        net.train()
        train_loss = []
        train_acc = []
        for (img, labels) in tqdm(train_iter):
            img = img.to(device)
            labels = labels.to(device)
            # 转发数据。(确保数据和模型在同一设备上）。

            processed_label = net(img)
            # 计算交叉熵损失。
            # 我们不需要在计算交叉熵之前应用 softmax，因为它会自动完成。
            l = loss(processed_label, labels)
            # 应首先清除上一步参数中存储的梯度。
            optimizer.zero_grad()
            # 计算的参数的梯度。
            l.backward()
            # 更新参数
            optimizer.step()
            # 计算当前批次的精度。
            sourceTensor = (processed_label.argmax(dim=-1) == labels)
            sourceTensor = sourceTensor.type(torch.float32)
            acc = torch.mean(sourceTensor)

            train_loss.append(l.item())
            train_acc.append(acc)

        # 训练集的平均损失和准确率是记录值的平均值。
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_acc) / len(train_acc)
        # 打印结果
        print(f"[ Train | {epoch + 1:03d}/{num_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- 验证 ----------
        # 确保模型处于评估模式，以便禁用某些模块（如 dropout）并使其正常工作。
        net.eval()
        valid_loss = []
        valid_acc = []
        for (img, labels) in tqdm(valid_iter):
            img = img.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                processed_label = net(img)
            l = loss(processed_label, labels)
            sourceTensor = (processed_label.argmax(dim=-1)) == labels
            sourceTensor = sourceTensor.type(torch.float32)
            acc = torch.mean(sourceTensor)
            valid_loss.append(l.item())
            valid_acc.append(acc)
        # 验证集的平均损失和准确率是记录值的平均值。
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_acc) / len(valid_acc)
        # 打印结果
        print(f"[ Valid | {epoch + 1:03d}/{num_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\n "
              f"----------------------------")
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(net.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc))


def test_model(net, test_iter, device):
    # 创建对应的model,并从训练处载入模型
    print(f'test on {device}')
    saveFileName = './submission.csv'
    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    predictions = []
    net.eval()
    for img in tqdm(test_iter):
        with torch.no_grad():
            labels = net(img.to(device))
        predictions.extend(labels.argmax(dim=-1).cpu().numpy().tolist())

    preds = []
    for item in predictions:
        preds.append(num2labels[item])

    test_data = pd.read_csv(test_path)
    test_data['label'] = pd.Series(preds)
    submission = pd.concat([test_data['image'], test_data['label']], axis=1)
    submission.to_csv(saveFileName, index=False)
    print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    # 生成训练,验证和测试数据集
    train_dataset = LeavesData(train_path, img_path, mode='Train')
    valid_dataset = LeavesData(train_path, img_path, mode='Valid')
    test_dataset = LeavesData(test_path, img_path, mode='Test')

    # 创建小批量抽样
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    # fig = plt.figure(figsize=(20, 12))
    # columns = 4
    # rows = 2
    # dataiter = iter(valid_loader)
    # inputs, classes = next(dataiter)
    #
    # for idx in range(columns * rows):
    #     ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    #     ax.set_title(num2labels[int(classes[idx])])
    #     plt.imshow(im_convert(inputs[idx]))
    # plt.show()

    model_path = './models/resnet34.ckpt'
    # 超参数
    resnet34 = res_mosdels(176)
    num_epochs = 20
    lr = 0.003

    # 训练模型
    train_model(resnet34, train_loader, valid_loader, num_epochs, lr, d2l.try_gpu())
    test_model(resnet34, test_loader, d2l.try_gpu())
