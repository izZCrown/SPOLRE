import matplotlib.pyplot as plt
import numpy as np

# The data to be used in the pie chart
totals = [18995, 13487, 3141]  # Total values for each category
breakdown = {  # Values for each system
    'omission': [5285, 3268, 1638, 1936, 2436, 2240, 2192],
    'misclassification': [1411, 1934, 1352, 1660, 2749, 1714, 2667],
    'numerical': [423, 401, 252, 305, 831, 448, 481]
}
def make_gradient(color, length):
    return [plt.cm.get_cmap(color)(0.7 * i / length) for i in range(length)]

# Create gradient colors for each section
colors1 = make_gradient('Reds', 14)
colors2 = make_gradient('Greens', 14)
colors3 = make_gradient('Blues', 14)

gradients = {
    'omission': colors1[6:-1],
    'misclassification': colors2[6:-1],
    'numerical': colors3[6:-1]
}
colors_inner = [colors1[-1], colors2[-1], colors3[-1]]

# Data for the outer circle
data_outer = breakdown['omission'] + breakdown['misclassification'] + breakdown['numerical']
colors_outer = gradients['omission'] + gradients['misclassification'] + gradients['numerical']

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 8), dpi=1000)
ax.set_position([0.2, 0.2, 0.6, 0.6])

# Outer circle: the systems
wedges, _ = ax.pie(data_outer, radius=1.5, colors=colors_outer, startangle=90, counterclock=False, wedgeprops=dict(width=0.6, edgecolor='w'))

# Inner circle: the total errors
wedges2, labels, pct_texts = ax.pie(totals, radius=0.9, colors=colors_inner, startangle=90, counterclock=False, autopct='%1.2f%%', pctdistance=0.75)
font_size = 9.5
font_color = '#383838'
font_family = 'monospace'
font_weight = 'normal'

for text in pct_texts:
    text.set_color(font_color)
    text.set_fontsize(12)
    text.set_fontfamily(font_family)
    text.set_fontweight = 'bold'

labels_outer = ['Azure', 'GIT', 'BLIP', 'BLIP2', 'VIT-GPT2', 'OFA', 'VinVL']*3

font_properties = {
    'family' : font_family,      # 字体类型，如 'serif', 'sans-serif', 'cursive', 'fantasy', 或 'monospace'
    'weight' : font_weight,       # 字体粗细，如 'normal', 'bold', 'light'
    'size'   : font_size,    # 字体大小
    'color'  : font_color       # 字体颜色
}

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1) / 2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    rotation = (ang + 90) % 180 - 90  # Correct the angle so the text is always upright
    text = f"{labels_outer[i]}: {data_outer[i]}"
    ax.text(x * 1.2, y * 1.2, text, ha='center', va='center', rotation=rotation,
            rotation_mode='anchor', fontdict=font_properties)

legend_labels = ['Omission', 'Misclassification', 'Numerical Inaccuracy']
legend_colors = [colors_inner[0], colors_inner[1], colors_inner[2]]

colors = []
colors += colors_inner
labels = []
labels += legend_labels
for i in range(len(gradients['omission'])):
    colors.append(gradients['omission'][i])
    colors.append(gradients['misclassification'][i])
    colors.append(gradients['numerical'][i])
    labels.append(' ' * len(labels_outer[i]))
    labels.append(labels_outer[i])
    labels.append(' ' * len(labels_outer[i]))

legend = ax.legend(handles=[plt.matplotlib.patches.Patch(facecolor=colors[i], label=labels[i]) 
                    for i in range(len(colors))], 
          loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=8)

for text in legend.get_texts():
    text.set_color(font_color)
    text.set_fontsize(10)
    text.set_fontfamily(font_family)
    text.set_fontweight = 'bold'



# Set the aspect ratio to be equal
ax.set_aspect('equal')

# Display the pie chart
plt.show()

# plt.tight_layout()

plt.savefig('/home/wgy/multimodal/MuMo/test/chart.pdf')
# plt.savefig('/home/wgy/multimodal/MuMo/test/chart.png')
plt.close()