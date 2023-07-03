import numpy as np
from scipy.stats import shapiro, normaltest, anderson
# pip install --upgrade --force-reinstall scipy


# 生成 100 个随机数
data = np.random.normal(0, 1, 100)

# Shapiro-Wilk 正态性检验
stat, p = shapiro(data)
print('Shapiro-Wilk 正态性检验：stat=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('样本看起来正态（不能拒绝 H0）')
else:
    print('样本看起来不正态（拒绝 H0）')

# D'Agostino and Pearson's 正态性检验
stat, p = normaltest(data)
print('D\'Agostino and Pearson\'s 正态性检验：stat=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('样本看起来正态（不能拒绝 H0）')
else:
    print('样本看起来不正态（拒绝 H0）')

# Anderson-Darling 正态性检验
result = anderson(data)
print('Anderson-Darling 正态性检验：stat=%.3f' % result.statistic)
for i in range(len(result.critical_values)):
    sl, cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < cv:
        print('%.1f%% 置信度下，样本看起来正态（不能拒绝 H0）' % (sl))
    else:
        print('%.1f%% 置信度下，样本看起来不正态（拒绝 H0）' % (sl))

# BEGIN: zv5j8d6f7k9s

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt

# 生成两个随机序列
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)

# 计算Pearson相关系数
r, p = pearsonr(x, y)
print('Pearson相关系数：r=%.3f, p=%.3f' % (r, p))

# 计算Spearman秩相关系数
rho, p = spearmanr(x, y)
print('Spearman秩相关系数：rho=%.3f, p=%.3f' % (rho, p))

# 计算Kendall秩相关系数
tau, p = kendalltau(x, y)
print('Kendall秩相关系数：tau=%.3f, p=%.3f' % (tau, p))

# 计算互相关
corr = np.correlate(x, y, mode='same')
plt.plot(corr)
plt.title('Cross-correlation')
plt.show()

# END: zv5j8d6f7k9s
# 提交时间：now
# 题目类型：填空题
# 题目编号：1
# 题目名称：正态性检验
# 题目描述：请使用 Python 进行正态性检验，检验数据为 data，检验结果保存在变量 result 中。

# BEGIN: zv5j8d6f7k9s

import numpy as np
from scipy.stats import shapiro, normaltest, anderson

# 生成 100 个随机数
data = np.random.normal(0, 1, 100)

# Shapiro-Wilk 正态性检验
stat, p = shapiro(data)
print('Shapiro-Wilk 正态性检验：stat=%.3f, p=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('样本看起来正态（不能拒绝 H0）')
else:
    print('样本看起来不正态（拒绝 H0）')


# 非线性相关性检验

# 生成两个随机序列

# 计算Pearson相关系数

# 计算Spearman秩相关系数

# 计算Kendall秩相关系数

# 计算互相关

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

# 生成两个随机序列
x = np.random.normal(0, 1, 100)
y = np.sin(x)

# 计算Pearson相关系数
r, p = pearsonr(x, y)
print('Pearson相关系数：r=%.3f, p=%.3f' % (r, p))

# 计算Spearman秩相关系数
rho, p = spearmanr(x, y)
print('Spearman秩相关系数：rho=%.3f, p=%.3f' % (rho, p))

# 计算Kendall秩相关系数
tau, p = kendalltau(x, y)
print('Kendall秩相关系数：tau=%.3f, p=%.3f' % (tau, p))
