from torch import nn
import torch
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=10, embedding_dim=16)  # 接收的分类的ID，即0到9
        # 的数字，是一个比较简单的结构化的数据，并不需要做tokenizer，所以这里进来就直接进行embedding。
        self.dense1 = nn.Linear(in_features=16, out_features=64)  # 向量从宽度16经过拉宽，变成64
        self.dense2 = nn.Linear(in_features=64, out_features=16)  # 再从64降维为16
        self.wt = nn.Linear(in_features=16, out_features=8)  # 再经过一层降低到8，最终输出的是宽度为8的文本向量
        self.ln = nn.LayerNorm(8)

    def forward(self, x):
        x = self.emb(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.wt(x)
        x = self.ln(x)
        return x


if __name__ == '__main__':
    text_encoder = TextEncoder()
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    y = text_encoder(x)
    print(y.shape)