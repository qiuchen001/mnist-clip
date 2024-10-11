import torch
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

model = CLIP().to(DEVICE)  # 模型

try:  # 加载模型
    model.load_state_dict(torch.load('model.pth', weights_only=True))
except:
    pass

optimzer = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器

ITER_BATCH_COUNT = 100000  # 迭代次数
BATCH_SIZE = 64  # 从batch内选出10个不一样的数字
TARGET_COUNT = 10  # 共10种数字


def train_model(dataloader):
    """
        训练模型
    """
    for i in range(ITER_BATCH_COUNT):
        while True:  # 确保一个批次中一定有10张图片，并且这10张图片一定是不重复的数字。避免导致模型脑裂，长时间的loss下不去
            imgs, labels = next(iter(dataloader))
            if torch.unique(labels).shape[0] < TARGET_COUNT:  # 未覆盖10种数字
                continue
            # 挑选出10个数字
            target = set()
            indexes = []
            for j in range(BATCH_SIZE):
                if labels[j].item() in target:
                    continue
                target.add(labels[j].item())
                indexes.append(j)
                if len(target) == TARGET_COUNT:
                    break
            imgs = imgs[indexes]
            labels = labels[indexes]
            break

        logits = model(imgs.to(DEVICE), labels.to(DEVICE))  # 拿到一批图片和分类ID，就可以调用clip模型，得到点积的矩阵

        targets = torch.arange(0, TARGET_COUNT).to(DEVICE)  # 生成文本和对应的图像的位置
        loss_i = F.cross_entropy(logits, targets)  # 之后做交叉熵的损失计算
        loss_t = F.cross_entropy(logits.permute(1, 0), targets)
        loss = (loss_i + loss_t) / 2

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if i % 1000 == 0:
            print('iter:{},loss:{}'.format(i, loss))
            torch.save(model.state_dict(), '.model.pth')
            os.replace('.model.pth', 'model.pth')


if __name__ == '__main__':
    # 在 Windows 上，需要调用 freeze_support()
    torch.multiprocessing.freeze_support()

    dataset = MNIST()  # 数据集

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                            persistent_workers=True)  # 数据加载器

    # 启动训练
    train_model(dataloader)
