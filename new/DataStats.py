# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:34:27 2021

@author: hasan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

df = pd.read_csv('Labels.csv')
male_norm = df.query('gender=="male" and label=="normal"')
male_abnm = df.query('gender=="male" and label=="abnormal"')
female_norm = df.query('gender=="female" and label=="normal"')
female_abnm = df.query('gender=="female" and label=="abnormal"')
n_bins = 21
d_labels = ['Normal', 'Abnormal']

'''-------------------Calculate Stats-------------------'''
mean_age = df.groupby(['gender']).mean()
std_age = df.groupby(['gender']).std()

mean_age_male = mean_age.loc['male']['age']
std_age_male = std_age.loc['male']['age']
n_male = len(male_norm) + len(male_abnm)
p_male_norm = (len(male_norm)/n_male)*100
p_male_abnm = 100 - p_male_norm

mean_age_female = mean_age.loc['female']['age']
std_age_female = std_age.loc['female']['age']
n_female = len(female_norm) + len(female_abnm)
p_female_norm = (len(female_norm)/n_female)*100
p_female_abnm = 100 - p_female_norm

p_male = (n_male/(n_male + n_female))*100
p_female = 100 - p_male

'''-------------------Plot Histograms------------------'''
bin_edges = np.linspace(start=0, stop = 100, num = n_bins, 
                         endpoint = True)
print(bin_edges)
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.hist([male_norm.age, male_abnm.age], bins=bin_edges,
           rwidth=0.85,  orientation='horizontal', stacked=True)
ax1.invert_xaxis()
ax1.yaxis.tick_right()
ax1.set_title(f'male ({p_male:0.2f}%)')
ax1.set_xlabel('count')
ax1.xaxis.set_label_coords(0.03, -0.05)
ax1.set_ylabel('age (years)')
ax2.hist([female_norm.age, female_abnm.age], bins=bin_edges,
           rwidth=0.85, orientation='horizontal', stacked=True, label=d_labels)
ax2.set_title(f'female ({p_female:0.2f}%)')
ax2.set_xlabel('count')
ax2.xaxis.set_label_coords(0.97, -0.05)

'''-----------------Center Align yticklabels-------------'''
y = [0, 20, 40, 60, 80, 100]
ylbls = y
ax1.set(yticks=y, yticklabels=[])
for ycoord, ylbl in zip(y, ylbls):
    ax1.annotate(ylbl, (0.51, ycoord), xycoords=('figure fraction', 'data'),
                     ha='center', va='center')

'''-----------------Plot Mean line & Std-Dev ---------------------'''
xLims = ax1.get_xlim()
ax1.plot(xLims, [mean_age_male, mean_age_male], 'k-', lw=2)
ax1.legend([f'Mean age: {mean_age_male:.2f}(\u00B1 {std_age_male:.2f})',
             f'Normal ({p_male_norm:.2f}%)', 
             f'Abnormal ({p_male_abnm:.2f}%)'],
             loc='lower center',  bbox_to_anchor=(0.5,-0.4))
'''(Command below) Since ax1 goes from left-2-right, the (xy) coordinate 
represents the bottom right corner of rect instead of the bottom left corner
in the Rectangle documentation'''
ax1.add_patch(Rectangle((0, mean_age_male - std_age_male), 
                        xLims[0], 2*std_age_male, color="grey", alpha=0.3))
ax1.set_xlim(xLims) #Gets rids of empty space between rect and Figure end
xLims = ax2.get_xlim()
ax2.plot(xLims, [mean_age_female, mean_age_female], 'k-', lw=2)
ax2.legend([f'Mean age: {mean_age_female:.2f}(\u00B1 {std_age_female:.2f})',
             f'Normal ({p_female_norm:.2f}%)', 
             f'Abnormal ({p_female_abnm:.2f}%)'],
             loc='lower center',  bbox_to_anchor=(0.5,-0.4))

ax2.add_patch(Rectangle((0, mean_age_female - std_age_female), 
                        xLims[1], 2*std_age_female, color="grey", alpha=0.3))
ax2.set_xlim(xLims) #Gets rids of empty space between rect and Figure end
plt.show()
fig.savefig('PopPyramid_img.png', format='png', dpi=1200, bbox_inches='tight')