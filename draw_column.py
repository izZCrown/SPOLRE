import matplotlib.pyplot as plt
import numpy as np

categories = ['Survey 1', 'Survey 2', 'Survey 3', 'Survey 4']
values1 = [5, 7, 3, 4]
values2 = [3, 6, 2, 5]

x = np.arange(len(categories))

plt.figure(figsize=(10, 6), dpi=1000)
font_family = 'Times New Roman'
font_weight = 'normal'
font_size = 12
font_color = '#383838'
font_properties = {
    'family' : font_family,      # 字体类型，如 'serif', 'sans-serif', 'cursive', 'fantasy', 或 'monospace'
    'weight' : font_weight,       # 字体粗细，如 'normal', 'bold', 'light'
    'size'   : font_size,    # 字体大小
    'color'  : font_color       # 字体颜色
}

bars1 = plt.bar(x - 0.15, values1, width=0.3, color='skyblue', edgecolor='black', label='Generated Images')
bars2 = plt.bar(x + 0.15, values2, width=0.3, color='orange', edgecolor='black', label='Real-world Images')

for bar in bars1 + bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, yval, ha='center', va='bottom', fontdict=font_properties)

title = plt.title('User Survey Result', fontsize=20, fontweight='bold')
title.set_fontfamily(font_family)
plt.xlabel('', fontsize=12, fontweight='bold')
plt.ylabel('', fontsize=12, fontweight='bold')


plt.xticks(x, categories, fontsize=15, fontweight='bold')
plt.yticks(fontsize=15, fontweight='bold')
ax = plt.gca()
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')



legend = plt.legend(frameon=True, shadow=True, borderpad=1)
for text in legend.get_texts():
    text.set_color('#383838')
    text.set_fontsize(12)

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()

plt.savefig('/home/wgy/multimodal/MuMo/test/column.png')
plt.close()
