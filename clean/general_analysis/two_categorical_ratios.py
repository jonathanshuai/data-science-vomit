"""Compare ratios for two categorical variables. Example output:
afridev                    functional                 0.677966
                           functional needs repair    0.023729
                           non functional             0.298305
cemo                       functional                 0.500000
                           functional needs repair    0.100000
                           non functional             0.400000
category1...               category2...               ratios sum to 1
"""


category1 = 'extraction_type'
category2 = 'label'

df[category1].value_counts()
compare_ratio_series = df.groupby([category1, category2])[category2].count()
compare_ratio_series = compare_ratio_series / compare_ratio_series.groupby(level=[0]).sum()


# If you want to style the ratios:
def ratio_style(ratio_series):
    styles = []

    # The safest way to style is to iterate w/ an index to
    # make sure you do something for each element. If more smart
    # you can do your own fancy thing too.
    for i in range(ratio_series.shape[0]):
        # If there's overwhelming functional ratio, color green
        if ratio_series.index[i][1] == 'functional':
            if ratio_series[i] > 0.60:
                styles.append('background-color: #58D68D')
            else:
                styles.append('')
        # If there's overwhelming nonfunctional ratio, color red
        elif ratio_series.index[i][1] == 'non functional':
            if ratio_series[i] > 0.60:
                styles.append('background-color: #EC7063')
            else:
                styles.append('')
        # If there's overwhelming need repair ratio, color ornage
        else:
            if ratio_series[i] > 0.30:
                styles.append('background-color: #F5B041')
            else:
                styles.append('')
                
    return styles

ratio_df.style.apply(ratio_style)
