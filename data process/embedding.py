import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np

# 固定随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)

# 读取数据
data = pd.read_excel(r"D:\hsx\desktop\machine learning-literature\em测试.xlsx")

# 定义处理函数
def process_cell(cell_value):
    if isinstance(cell_value, str):
        # 将字符串分割为浮点数列表
        numbers = [float(i) for i in cell_value.split(',') if i]  # 忽略空字符串
    else:
        # 处理非字符串的情况，返回空列表
        numbers = []
    return numbers

# 处理每个单元格
matrix_column = [process_cell(value) for value in data.iloc[:, 2]]

# 去掉长度为0的序列（如果存在）
matrix_column = [seq for seq in matrix_column if len(seq) > 0]

# 计算最大长度（用于填充）
max_length = max(len(seq) for seq in matrix_column)
print(f"Maximum length of sequences: {max_length}")

# 找到最小的浮点数数量
min_float_count = min(len(seq) for seq in matrix_column if len(seq) > 0)
print(f"Minimum number of floats in a sequence: {min_float_count}")

# 定义嵌入网络
class MatrixEmbedder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(MatrixEmbedder, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed_input = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, _) = self.lstm(packed_input)
        output = self.fc(hn[-1])
        return output

# 数据准备：将列表转换为张量
# 填充序列到最大长度
padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in matrix_column]
lengths = [len(seq) for seq in matrix_column]  # 实际长度

# 确保长度都大于0
lengths = [l for l in lengths if l > 0]

# 将填充后的序列和长度转换为张量
matrix_tensors = torch.tensor(padded_sequences, dtype=torch.float32).unsqueeze(-1)  # 增加最后一个维度以适配LSTM输入
lengths_tensor = torch.tensor(lengths, dtype=torch.int64)

# 检查是否有长度为0的样本
if len(lengths_tensor) == 0:
    raise ValueError("No valid sequences found after filtering.")

# 初始化模型
hidden_dim = min_float_count  # 设置为最小的浮点数数量
output_dim = 1
model = MatrixEmbedder(hidden_dim, output_dim)

# 计算嵌入
embeddings = model(matrix_tensors, lengths_tensor)

# 打印嵌入结果
print(embeddings)

# 将嵌入的结果转换为 pandas DataFrame
embedding_df = pd.DataFrame(embeddings.detach().numpy(), columns=['embedding'])

# 将结果保存为 Excel 文件
embedding_df.to_excel(r"D:\hsx\desktop\em.xlsx", index=False)  # 替换为你想保存的路径