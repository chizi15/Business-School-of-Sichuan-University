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