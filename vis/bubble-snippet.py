# Check the difference between categories of provisions.
# This is defined by the mean of the mean differences in 
# (2017) gun homicide rates for each provision in the provision category.

# i.e. we group the results from provision comparisons by category and take the mean.
# We can also average over all years (2000 - 2017), but here I'll just focus on 2017.

# Another idea would be to use a slider animation from plotly, if I can find 
# an easy way to incorporate an iframe in a presentation/report.

# Select categories to highlight, and ones to exclude from the visual 
highlight = ['Allow local government to regulate', 'Buyer regulations']

to_exclude = ['Ammunition regulations' , 
              'Prohibitions for high-risk gun possession',
              'Possession regulations',
              'Stand your ground']

results = compare_provisions(annual_df, provisions_df, threshold=0, year=2017)    
results['n_with'] = states.shape[0] - results['n_without'] # Get n of states with provision
results = pd.merge(results, provisions_cat_df)

grouped_means = results.groupby('category')[['with', 'without', 'n_with']].mean()
grouped_means['count'] = results.groupby('category')['with'].count().values
grouped_means = grouped_means.sort_values('count')
grouped_means = grouped_means[~grouped_means.index.isin(to_exclude)]

categories = grouped_means.index.values
x = grouped_means['n_with'].values
y = grouped_means['count'].values

# Apply jitter 
def apply_jitter(x, y, x_tol=2, y_tol=2, x_jitter=0, y_jitter=1, axis_jitter=1.5):
    x_jittered = []
    y_jittered = []
    n_points = 0
    for point_x, point_y in zip(x, y):
        for i in range(n_points):
            too_close_x = abs(x_jittered[i] - point_x) <= x_tol
            too_close_y = abs(y_jittered[i] - point_y) <= y_tol
            if too_close_x and too_close_y:
                if point_x < x_jittered[i]:
                    x_jittered[i] += x_jitter
                    point_x -= x_jitter
                else:
                    x_jittered[i] -= x_jitter
                    point_x += x_jitter
                if point_y < y_jittered[i]:
                    y_jittered[i] += y_jitter
                    point_y -= y_jitter
                else:
                    y_jittered[i] -= y_jitter
                    point_y += y_jitter
        
        if point_x <= x_tol:
            point_x += axis_jitter
        if point_y <= y_tol:
            point_y += axis_jitter
            
        x_jittered.append(point_x)
        y_jittered.append(point_y)
        n_points += 1
    return x_jittered, y_jittered

# Plot the results
colors = [
    '#7FADD5',
    '#7F98D5',
    '#7F89D5',
    '#D57F82'
] 
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'bubbles', colors, N=100)

exag_pow = 1.7 # To exaggerate bubble sizes to easier compare them

x, y = apply_jitter(x, y)
z = grouped_means['with'] - grouped_means['without']
# c = z.apply(lambda x: cmap[0] if x < 0 else cmap[1])
z_abs = abs(z) ** exag_pow

# Let's make a pretty bubble plot this time to mix up our visualizations
bubble_scale = 3400

fig = plt.figure(figsize=(15, 12))
plt.scatter(x, y, 
            s=z_abs * bubble_scale, 
            c=z,
            cmap=cmap,
            alpha=0.3, 
            edgecolor='None')

# Also, let's highlight the interesting groups 
grouped_means['x'] = x
grouped_means['y'] = y
grouped_means['z_abs'] = z_abs
highlighted_means = grouped_means.loc[highlight]
plt.scatter(highlighted_means['x'].values,
            highlighted_means['y'].values,
            s=highlighted_means['z_abs'].values * bubble_scale,
            facecolor='None',
            edgecolor='#7FBDC0',
            linewidth=3,
            alpha=0.6)


## Adding keys to make readable
base_x = 36.5
base_y = 0

# Size key: first get units in pixels
ax = plt.gca()
units = ax.transData.transform([(0,1),(1,0)]) - ax.transData.transform((0,0))
[[_,y_unit],[x_unit,_]] = units

# Then get the relative positions of each circle, and plot them
scales = np.array([0.5, 1.5, 3]) 
scales_transform = scales ** exag_pow
small_y = base_y + (np.sqrt(bubble_scale * scales_transform[0] / np.pi)) / y_unit
medium_y = base_y + (np.sqrt(bubble_scale * scales_transform[1] / np.pi)) / y_unit
large_y = base_y + (np.sqrt(bubble_scale * scales_transform[2] / np.pi))/ y_unit
plt.scatter([base_x, base_x, base_x], 
            [small_y, medium_y, large_y], 
            s=bubble_scale * scales_transform, 
            c=colors[0], alpha=0.25, clip_on=False)

# Color key: set coordinates and then plot two circles; one for each color!
color_key_x = base_x - 3
color_key_y1 = base_y + 12.5 # This one is on top
color_key_y2 = base_y + 10

plt.scatter([color_key_x, color_key_x],
            [color_key_y1, color_key_y2],
            s=[bubble_scale * 0.2, bubble_scale * 0.2],
            c=[colors[0], colors[-1]],
            alpha=0.3, clip_on=False)

# Set the limits of the plot along with labels and padding
plt.xlim(0, 32)
plt.ylim(0, 25)
plt.title('Effects of Provision Categories on Gun Violence in 2017\n', size=24, color='#434343')
plt.xlabel('Prevalence of the Provision Category*', size=12, labelpad=20)
plt.ylabel('Number of Provisions in Category', size=12, labelpad=20)
plt.gca().tick_params(axis='both', which='major', pad=10)

# Add annotations for bubbles
for i, txt in enumerate(categories):
    plt.annotate(txt, (x[i],y[i]), ha='center', color='#434343', fontsize=11)

# Add size key annotations
for (txt, y_coord) in zip(scales, [small_y, medium_y, large_y]):
    # Move y_coord one radius up; and subtract some padding
    radius = (y_coord - base_y)
    y_coord += radius - 0.3
    plt.annotate('{:.1f}'.format(txt), (base_x, y_coord), 
                 va='top', ha='center', color='#434343', 
                 fontsize=12, annotation_clip=False)

# Add the title of the size key (we just add some padding to the final y_coord)
y_coord += 1.4
plt.annotate('Difference in gun violence', (base_x, y_coord), va='bottom', ha='center', 
                 color='#434343', fontsize=14, annotation_clip=False)

# Add color key labels:
plt.annotate('Gun violence decreased\nwith category', (color_key_x + 0.8, color_key_y1), 
                 va='center', ha='left', color='#434343', 
                 fontsize=14, annotation_clip=False)
plt.annotate('Gun violence increased\nwith category', (color_key_x + 0.8, color_key_y2), 
                 va='center', ha='left', color='#434343', 
                 fontsize=14, annotation_clip=False)
    
apply_styling()
plt.show()
