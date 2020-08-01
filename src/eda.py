import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.configuration as config



def count_each_unique(lst):
    """
    Count occurences of each unique element of a list
    Args:
        lst (list): list or other iterable object
    Returns:
        pandas.DataFrame() object with fields 'element', 'count', and 'percent'
    """
    unique_values = list(set(lst))
    uv_counts = [len([l for l in lst if l == uv]) for uv in unique_values]
    uv_percent_counts = [uvc / sum(uv_counts) for uvc in uv_counts]
    output_df = pd.DataFrame({'element' : unique_values,
                              'count' : uv_counts,
                              'percent' : uv_percent_counts})
    return output_df


    
def plot_frequency_counts(lst, xlab = 'Element', ylab = 'Frequency',
                          title = 'Frequency Counts', figsize = (9, 6),
                          percentage_decimals = 2, color = 'seagreen', alpha = 0.3):
    """
    Wrapper around count_each_unique() function to create and print matplotlib bar plot 
    Args:
        lst (list): list or other iterable object
        xlab (str): x-axis label. defaults to 'Element'
        ylab (str): y-axis label. defaults to 'Frequency'
        title (str): plot title. defaults to 'Frequency Counts'
        figsize (tuple): x and y axis dimensions for printed plot
        percentage_decimals (int): number of decimal places in % label on each bar. defaults to 2.
        color (str): color used for bars and text labels. defaults to 'seagreen'.
        alpha (float): transparency (float zero to onne). defaults to 0.3.
    """
    # Generate Frequency Counts & Percentage Labels
    freq_count_df = count_each_unique(lst)
    freq_count_df['percent_lab'] = [f'{round(x * 100, percentage_decimals)} %' for x in freq_count_df['percent']]
    
    # Create Basic Bar Plot
    plt.figure(figsize = figsize)
    ax = freq_count_df['count'].plot(kind = 'bar', color = color, alpha = alpha, edgecolor = color)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xticklabels(freq_count_df['element'])
    #plt.show()
    
    # Create Labels
    rects = ax.patches
    (y_min, y_max) = ax.get_ylim()
    y_height = y_max - y_min
    for i, r in enumerate(rects):
        height = r.get_height()
        label_txt = str('%d' % int(height)) + ' (' + freq_count_df['percent_lab'][i] + ')'
        label_pos = height + (y_height * 0.01)
        ax.text(r.get_x() + r.get_width() * 0.5, label_pos,
                label_txt, ha = 'center', va = 'bottom', color = color)
    plt.show()