# importing the used packages 
import pandas as pd
import numpy as np
from IPython.display import display

# importing package to create plots and setting basic, visual settings
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"-"})
plt.rcParams.update({'font.size': 10})
import ipywidgets as widgets

def plot_priceindex(df, selected_provinces):
    fig, ax = plt.subplots()
    for province in selected_provinces:
        I = df['PROVINCE'] == province
        df.loc[I, :].plot(x='TIME', y='SALES_INDEX', legend=False, ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price Index')
    ax.set_title('Price Index of Houses Across Regions in Denmark')
    plt.legend(selected_provinces)
    plt.show()

def priceindex_widgets(df):
    widgets.interact(plot_priceindex, # creating interactive widget letting us choose the desired provinces
                 df=widgets.fixed(df),
                 selected_provinces=widgets.SelectMultiple(description='Provinces', 
                                                           options=df.PROVINCE.unique(), 
                                                           value=['Province Byen København'], disabled=False))

def _binscatter(df,province):
    # Construct dataset for each province
    I = df.loc[df['PROVINCE'] == province,:].copy()

    # define bin width, such that we discretize UNEMPLOYMENT_RATE in 10 equal-sized bins
    bin_width = (np.max(I['UNEMPLOYMENT_RATE'])-np.min(I['UNEMPLOYMENT_RATE']))/10.0

    # assign each province in each year to its relevant bin
    I['unemp_bin'] = np.ceil(I['UNEMPLOYMENT_RATE']/bin_width)*bin_width

    # collapse dataset to mean of SALES_INDEX within each bin
    collapsed = I.groupby(['unemp_bin'], dropna=True)['SALES_INDEX'].apply('mean')

    # Create figure
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel('Sales index')
    ax.set_xlabel('Unemployment rate (%)')
    ax.set_title('Correlation between unemployment rate and house prices within provinces')

    # Plot binned scatterplot
    ax.scatter(collapsed.index, collapsed.values, alpha=0.7, edgecolors="k")

    # Plot linear regression line (a is slope and b is intercept)
    a, b = np.polyfit(collapsed.index, collapsed.values, deg=1)

    # create sequence of 100 numbers from minimum to maximum of bins
    xseq = np.linspace(np.min(collapsed.index),np.max(collapsed.index),100)
    
    # Plot regression line
    ax.plot(xseq, b + a * xseq, color="red", linestyle="--", lw=2.5, alpha=0.7)

    # Add slope as text
    x, y = collapsed.index[5], collapsed.values[5]
    ax.annotate(f'slope={a:.3f}',xy=(x,y+8), xytext=(x, -x), textcoords='offset points');


def binscatter_widgets(df):
    widgets.interact(_binscatter, # creating interactive widget letting us choose the desired province
        df = widgets.fixed(df),
        province = widgets.Dropdown(description='Province', 
                                        options=df.PROVINCE.unique(), 
                                        value='Province Byen København'),
    );
