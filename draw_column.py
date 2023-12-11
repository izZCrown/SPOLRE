import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
values1 = [5, 7, 3, 4]
values2 = [3, 6, 2, 5]

# X轴位置
x = np.arange(len(categories))

# 绘图
plt.bar(x - 0.2, values1, 0.4, label='Value 1')
plt.bar(x + 0.2, values2, 0.4, label='Value 2')

# 标签
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.xticks(x, categories)
plt.legend()

plt.savefig('/home/wgy/multimodal/MuMo/test/column.pdf')
plt.close()
