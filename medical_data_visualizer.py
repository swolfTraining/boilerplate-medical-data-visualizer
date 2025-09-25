import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / (df['height']/100)**2) // 25
df.loc[df['overweight'] > 1, 'overweight'] = 1

# 3
df.loc[df['cholesterol'] <= 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] <= 1, 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc'] = 1

# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(['cardio'], ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count()
    df_cat = df_cat.to_frame()
    df_cat = df_cat.rename(columns={'value': 'total'})
    df_cat = df_cat.reset_index()

    # 7
    df_cat_plot = sns.catplot(df_cat, x = 'variable', y='total', col = 'cardio', hue= 'value', kind = 'bar')

    # 8
    fig = df_cat_plot.figure

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[df['ap_lo'] <= df['ap_hi']]
    df_heat = df_heat[df_heat['height'] >= df['height'].quantile(0.025)]
    df_heat = df_heat[df_heat['height'] <= df['height'].quantile(0.975)]
    df_heat = df_heat[df_heat['weight'] >= df['weight'].quantile(0.025)]
    df_heat = df_heat[df_heat['weight'] <= df['weight'].quantile(0.975)]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.zeros(corr.shape)
    mask[np.triu_indices(len(mask))] = 1
    mask = mask.astype(bool)
    
    # 14
    fig, ax = plt.subplots()

    # 15
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='.1f')

    # 16
    fig.savefig('heatmap.png')
    return fig
