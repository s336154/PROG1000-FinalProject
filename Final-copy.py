# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:20:02 2022

@author: saris
"""

'''
1)
a) Import the file "sp500_marketvalues.csv".
b) Import the file "sp500_survey.csv"
c) Merge the two data sets. Use "Symbol" as a common column.
d) Drop any NaNs.
e) Create a new variable called "returns" which is the yearly stock return of 
each firm in the data from April 2017 to April 2022. The formula for the 
returns is:
𝑟𝑒𝑡𝑢𝑟𝑛𝑠 = (
𝐹𝑖𝑟𝑚𝑉2022
𝐹𝑖𝑟𝑚𝑉2017)
1
5
-


'''

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

###############################################################################
################################### 1.(a) #####################################
###############################################################################
values = pd.read_csv('sp500_marketvalues.csv')

###############################################################################
################################### 1.(b) #####################################
###############################################################################
survey = pd.read_csv('sp500_survey.csv')

###############################################################################
################################### 1.(c) #####################################
###############################################################################
values_survey = values.merge(survey, on='Symbol', how='outer')

###############################################################################
################################### 1.(d) #####################################
###############################################################################
values_survey.dropna(inplace=True)
# s= values_survey.isna().sum()

###############################################################################
################################### 1.(e) #####################################
###############################################################################
F𝑖𝑟𝑚𝑉2022 = values_survey['FirmV2022']
F𝑖𝑟𝑚𝑉2017 = values_survey['FirmV2017']

returns_lst = []
for item1, item2 in zip(values_survey['FirmV2022'], values_survey['FirmV2017']):
    ret = ((item1 / item2) ** (1 / 5)) - 1
    returns_lst.append(ret)

values_survey['returns'] = returns_lst
# values_survey.to_excel('values_survey.xlsx', sheet_name = 'SurveyValues', index = False)

'''

2) Explore the data set.
a) Present descriptive statistics for the data set.
b) Create a figure with four sub-plots in two rows and two columns. "returns" 
should be on the y-axis and differentiation, cost leadership, efficiency, and 
novelty should be on the x-axis. Add the correlation coefficients in the 
subplot titles.

'''

###############################################################################
################################### 2.(a) #####################################
###############################################################################

vs_2 = values.merge(survey, on='Symbol', how='outer')

vs_2.dropna(inplace=True)

F𝑖𝑟𝑚𝑉2022 = vs_2['FirmV2022']
F𝑖𝑟𝑚𝑉2017 = vs_2['FirmV2017']

vs_returns = []
for item1, item2 in zip(vs_2['FirmV2022'], vs_2['FirmV2017']):
    ret = ((item1 / item2) ** (1 / 5)) - 1
    vs_returns.append(ret)

vs_2['returns'] = vs_returns

if 'Sector' in vs_2:
    vs_2.drop('Sector', axis=1, inplace=True)
if 'Name_x' in vs_2:
    vs_2.drop('Name_x', axis=1, inplace=True)
if 'Name_y' in vs_2:
    vs_2.drop('Name_y', axis=1, inplace=True)

print('\n\n===================================================================')
print('\r==================== Exercise 2.a =================================')

print('\n\n >> Here are descriptive statistics for the data set: ')
print('\n >> The total number of documented values for each variable is: \n')
print(vs_2.count())
print('\n >> The mean outcome for each variable in the data set is: \n')
print(vs_2.mean())
print('\n >> The maximum value for each variable in the data set is: \n')
print(vs_2.max())
print('\n >> The minimum value for each variable in the data set is: \n')
print(vs_2.max())
print('\n >> The standard deviation for each variable in the data set is: \n')
print(vs_2.std())

###############################################################################
################################### 2.(b) #####################################
###############################################################################

Y = values_survey['returns']
X1 = values_survey['Differentiation']
X2 = values_survey['Cost leadership']
X3 = values_survey['Efficiency']
X4 = values_survey['Novelty']

# Xc1 = sm.add_constant(X1)
# Xc2 = sm.add_constant(X2)
# Xc3 = sm.add_constant(X3)
# Xc4 = sm.add_constant(X4)

corr = values_survey.corr()
corr_def = values_survey.corr().loc['returns', 'Differentiation']
corr_cosled = values_survey.corr().loc['returns', 'Cost leadership']
corr_effec = values_survey.corr().loc['returns', 'Efficiency']
corr_nov = values_survey.corr().loc['returns', 'Novelty']

# create figure and subplots
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# plot subplots
ax[0, 0].scatter(X1, Y,
                 marker='o',
                 s=40,
                 color='green')
ax[0, 1].scatter(X2, Y,
                 marker='o',
                 s=40,
                 color='lightblue')
ax[1, 0].scatter(X3, Y,
                 marker='o',
                 s=40,
                 color='purple')
ax[1, 1].scatter(X4, Y,
                 marker='o',
                 s=40,
                 color='orange')

# set axis labels
ax[0, 0].set_ylabel('Returns', fontsize=17)
ax[1, 0].set_ylabel('Returns', fontsize=17)
ax[0, 1].set_ylabel('Returns', fontsize=17)
ax[1, 1].set_ylabel('Returns', fontsize=17)

ax[0, 0].set_xlabel('Differentiation', fontsize=17)
ax[0, 1].set_xlabel('Cost leadership', fontsize=17)
ax[1, 0].set_xlabel('Efficiency', fontsize=17)
ax[1, 1].set_xlabel('Novelty', fontsize=17)

# set axis range
ax[0, 0].set_ylim(min(Y), max(Y))
ax[1, 0].set_ylim(min(Y), max(Y))
ax[0, 1].set_ylim(min(Y), max(Y))
ax[1, 1].set_ylim(min(Y), max(Y))

ax[0, 0].set_xlim(min(X1), max(X1))
ax[0, 1].set_xlim(min(X2), max(X2))
ax[1, 0].set_xlim(min(X3), max(X3))
ax[1, 1].set_xlim(min(X4), max(X4))

# add title to figure
fig.suptitle('Market values survey', fontsize=30)

# add subplot title
ax[0, 0].set_title('r = ' + str(round(corr_def, 2)))
ax[0, 1].set_title('r = ' + str(round(corr_cosled, 2)))
ax[1, 0].set_title('r = ' + str(round(corr_effec, 2)))
ax[1, 1].set_title('r = ' + str(round(corr_nov, 2)))

# hide the right and top spines
ax[0, 0].spines[['right', 'top']].set_visible(False)
ax[0, 1].spines[['right', 'top']].set_visible(False)
ax[1, 0].spines[['right', 'top']].set_visible(False)
ax[1, 1].spines[['right', 'top']].set_visible(False)

# save figure
fig.savefig('values_survey.png',  # name (and path) for storing image
            dpi=300,  # image resolution
            bbox_inches='tight')  # remove white space around image

'''
3)
a) Create a bar plot to show the average yearly returns for each sector.
b) Create a plot with three subplots, one for each of the sectors. The subplots 
should be a histogram of the yearly returns within each sector.

'''
###############################################################################
################################### 3.(a) #####################################
###############################################################################


x = values_survey['Sector'].unique()
y = round(100 * values_survey.groupby(['Sector'])['returns'].mean(), 2)
# y = f"{y} %"


fig, ax = plt.subplots()

ax.bar(x, y, 0.5,
       facecolor='lightblue',
       edgecolor='blue',
       alpha=1,
       hatch='/')

# set axis label
ax.set_ylabel('Returns', color='blue', fontsize=16)
ax.set_xlabel('Sectors', color='blue', fontsize=16)

# set figure title
ax.set_title('Average sectors returns', fontsize=24)

# hide the right and top spines
ax.spines[['right', 'top']].set_visible(False)

# save figure
plt.savefig('barSectors.png',  # name (and path) for storing image
            dpi=300,  # image resolution
            bbox_inches='tight')  # remove white space around image

###############################################################################
################################### 3.(b) #####################################
###############################################################################
# create figure and subplots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 12))

# plot subplots
ax[0].hist(values_survey[values_survey['Sector'] == 'Industrials']['returns'].dropna(), bins=20,
           facecolor='green',
           edgecolor='black',
           alpha=1,
           label='Industrial Sector')

ax[1].hist(values_survey[values_survey['Sector'] == 'Information Technology']['returns'].dropna(), bins=20,
           facecolor='lightblue',
           edgecolor='blue',
           alpha=1,
           label='Information Technology Sector')

ax[2].hist(values_survey[values_survey['Sector'] == 'Financials']['returns'].dropna(), bins=20,
           facecolor='pink',
           edgecolor='purple',
           alpha=1,
           label='Financial Sector')

# set axis label
ax[0].set_xlabel('Returns', color='green', fontsize=16)
ax[0].set_ylabel('Total Occurence', color='green', fontsize=16)

ax[1].set_xlabel('Returns', color='blue', fontsize=16)
ax[1].set_ylabel('Total Occurence', color='blue', fontsize=16)

ax[2].set_xlabel('Returns', color='purple', fontsize=16)
ax[2].set_ylabel('Total Occurence', color='purple', fontsize=16)

# set figure title
fig.suptitle('Sectors returns', fontsize=27)

# add subplot title
ax[0].set_title('Industrial Sector', fontsize=17)
ax[1].set_title('Information Technology Sector', fontsize=17)
ax[2].set_title('Financial Sector', fontsize=17)

# hide the right and top spines
ax[0].spines[['right', 'top']].set_visible(False)
ax[1].spines[['right', 'top']].set_visible(False)
ax[2].spines[['right', 'top']].set_visible(False)

# save figure
plt.savefig('histSectors.png',  # name (and path) for storing image
            dpi=300,  # image resolution
            bbox_inches='tight')  # remove white space around image

'''

4)
a) Estimate the regression model: 
𝑟𝑒𝑡𝑢𝑟𝑛𝑠 = 𝛼 + 𝛽1𝐷𝑖𝑓𝑓𝑒𝑟𝑒𝑛𝑡𝑖𝑎𝑡𝑖𝑜𝑛 + 𝛽2𝐶𝑜𝑠𝑡 𝑙𝑒𝑎𝑑𝑒𝑟𝑠ℎ𝑖𝑝 + 𝛽3𝐸𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑐𝑦
+ 𝛽4𝑁𝑜𝑣𝑒𝑙𝑡𝑦 + 𝜖
How much of the variation in returns does the model explain?

'''
###############################################################################
################################### 4.(a) #####################################
###############################################################################

X = values_survey[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]  # choose two explanatory variables
Y = values_survey['returns']

# add constant
X = sm.add_constant(X)

# fit model
model = sm.OLS(Y, X).fit()

# model summary
mod = model.summary()

# model parameters
alpha = round(model.params['const'], 5)
beta1 = round(model.params['Differentiation'], 5)
beta2 = round(model.params['Cost leadership'], 5)
beta3 = round(model.params['Efficiency'], 5)
beta4 = round(model.params['Novelty'], 5)

# creating prediction variable for the returns
pred = model.predict(X)
values_survey['Pred'] = pred

# calculating residuals in the model
values_survey['Residual'] = values_survey['returns'] - values_survey['Pred']
res = round(values_survey['Residual'].sum(), 2)

# Differentiation= None; Cost_leadership= None; Efficiency=None; Novelty=None;
returns = str(alpha) + " +" + str(beta1) + " Differentiation  +" + str(beta2) + " Cost \
leadership +" + str(beta3) + " Efficiency +" + str(beta4) + " Novelty"

if "+-" in returns:
    returns = returns.replace("+-", "-")

adj_rsq = model.rsquared_adj
print('\n\n===================================================================')
print('\r==================== Exercise 4 a =================================')

# print('\n\n(*) Here is an equation that explains relationship between returns and \nDifferentiation, Cost leadership, Efficiency and Novelty: ')
print('\n\n >> Here is an estimate for the regression model: ')
print('\nReturns = ' + returns)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(100 * adj_rsq))

"""
b) If a company in the financial sector has a score of:
Differentiation: 54
Cost leadership: 61
Efficiency: 57
Novelty: 42
What would be the expected returns?
"""
###############################################################################
################################### 4.(b) #####################################
###############################################################################

print('\n\n===================================================================')
print('\r======================== Exercise 4 b =============================')
# print('\n\n(*) Inorder to estimate the according return value type in the values for \nthe following variables: \n\n')


Differentiation = 54  # int(input('Differentiation: '))
Cost_leadership = 61  # int(input('Cost leadership: '))
Efficiency = 57  # int(input('Efficiency: '))
Novelty = 42  # int(input('Novelty: '))

returns = alpha + beta1 * Differentiation + beta2 * Cost_leadership + beta3 * Efficiency \
          + beta4 * Novelty

print('\n\nReturns = {:.5f} (i.e. almost {} %)'.format(returns, round(100 * returns, 2)))

###############################################################################
################################### 4.(c) #####################################
###############################################################################

print('\n\n===================================================================')
print('\r======================== Exercise 4 c =============================')
# print('\n\n(*) Inorder to estimate the according return value type in the values for \nthe following variables: \n\n')


Differentiation = 54  # int(input('Differentiation: '))
Cost_leadership = 61  # int(input('Cost leadership: '))
Efficiency = 57  # int(input('Efficiency: '))
Novelty = 55  # int(input('Novelty: '))

returns = alpha + beta1 * Differentiation + beta2 * Cost_leadership + beta3 * Efficiency \
          + beta4 * Novelty

print('\n\nReturns = {:.5f} (i.e. almost {} %)'.format(returns, round(100 * returns, 2)))
# print('\n\n====================================================================')
# print('\r====================================================================')

'''
5)
a) Split the data into three parts, one for each of the sectors (Financials,
Industrials, Information Technology)
'''
###############################################################################
################################### 5.(a) #####################################
###############################################################################


vs_Financials = values_survey[values_survey['Sector'] == "Financials"]
vs_Industrials = values_survey[values_survey['Sector'] == "Industrials"]
vs_IT = values_survey[values_survey['Sector'] == "Information Technology"]

"""
b) Run a regression on each of the sub-sets of data. Does any of these 
regressions obtain a higher adjusted 𝑅
2
than the regression in 4) a) which 
used the whole data set?

"""

###############################################################################
################################## 5.(b).1 ####################################
###############################################################################

X_2 = vs_Financials[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]  # choose two explanatory variables
Y_2 = vs_Financials['returns']

# add constant
X_2 = sm.add_constant(X_2)

# fit model
model2 = sm.OLS(Y_2, X_2).fit()

# model summary
mod2 = model2.summary()

# model parameters
alpha2 = round(model2.params['const'], 5)
beta1_2 = round(model2.params['Differentiation'], 5)
beta2_2 = round(model2.params['Cost leadership'], 5)
beta3_2 = round(model2.params['Efficiency'], 5)
beta4_2 = round(model2.params['Novelty'], 5)

# creating prediction variable for the returns
pred2 = model2.predict(X_2)
vs_Financials['Pred'] = pred2

# calculating residuals in the model
vs_Financials['Residual'] = vs_Financials['returns'] - vs_Financials['Pred']
res2 = round(vs_Financials['Residual'].sum(), 2)

# Differentiation= None; Cost_leadership= None; Efficiency=None; Novelty=None;
returns_2 = str(alpha2) + " +" + str(beta1_2) + " Differentiation  +" + str(beta2_2) + " Cost \
leadership +" + str(beta3_2) + " Efficiency +" + str(beta4_2) + " Novelty"

if "+-" in returns_2:
    returns_2 = returns_2.replace("+-", "- ")

adj_rsq2 = model2.rsquared_adj
print('\n\n===================================================================')
print('\r==================== Exercise 5.b.1 ===============================')

print('\n\n >> Here is an estimate for the regression model for Financial Sector: ')
print('\nReturns = ' + returns_2)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(100 * adj_rsq2))

###############################################################################
################################## 5.(b).2 ####################################
###############################################################################


X_3 = vs_Industrials[
    ['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]  # choose two explanatory variables
Y_3 = vs_Industrials['returns']

# add constant
X_3 = sm.add_constant(X_3)

# fit model
model3 = sm.OLS(Y_3, X_3).fit()

# model summary
mod3 = model3.summary()

# model parameters
alpha3 = round(model3.params['const'], 5)
beta1_3 = round(model3.params['Differentiation'], 5)
beta2_3 = round(model3.params['Cost leadership'], 5)
beta3_3 = round(model3.params['Efficiency'], 5)
beta4_3 = round(model3.params['Novelty'], 5)

# creating prediction variable for the returns
pred3 = model3.predict(X_3)
vs_Industrials['Pred'] = pred3

# calculating residuals in the model
vs_Industrials['Residual'] = vs_Industrials['returns'] - vs_Industrials['Pred']
res3 = round(vs_Industrials['Residual'].sum(), 2)

# Differentiation= None; Cost_leadership= None; Efficiency=None; Novelty=None;
returns_3 = str(alpha3) + " +" + str(beta1_3) + " Differentiation  +" + str(beta2_3) + " Cost \
leadership +" + str(beta3_3) + " Efficiency +" + str(beta4_3) + " Novelty"

if "+-" in returns_3:
    returns_3 = returns_3.replace("+-", "- ")

adj_rsq3 = model3.rsquared_adj
print('\n\n===================================================================')
print('\r==================== Exercise 5.b.2 ===============================')

print('\n\n >> Here is an estimate for the regression model for Industrial Sector: ')
print('\nReturns = ' + returns_3)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(100 * adj_rsq3))

###############################################################################
################################## 5.(b).3 ####################################
###############################################################################


X_4 = vs_IT[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]  # choose two explanatory variables
Y_4 = vs_IT['returns']

# add constant
X_4 = sm.add_constant(X_4)

# fit model
model4 = sm.OLS(Y_4, X_4).fit()

# model summary
mod4 = model4.summary()

# model parameters
alpha4 = round(model4.params['const'], 5)
beta1_4 = round(model4.params['Differentiation'], 5)
beta2_4 = round(model4.params['Cost leadership'], 5)
beta3_4 = round(model4.params['Efficiency'], 5)
beta4_4 = round(model4.params['Novelty'], 5)

# creating prediction variable for the returns
pred4 = model4.predict(X_4)
vs_IT['Pred'] = pred4

# calculating residuals in the model
vs_IT['Residual'] = vs_IT['returns'] - vs_IT['Pred']
res4 = round(vs_IT['Residual'].sum(), 2)

# Differentiation= None; Cost_leadership= None; Efficiency=None; Novelty=None;
returns_4 = str(alpha4) + " +" + str(beta1_4) + " Differentiation  +" + str(beta2_4) + " Cost \
leadership +" + str(beta3_4) + " Efficiency +" + str(beta4_4) + " Novelty"

if "+-" in returns_4:
    returns_4 = returns_4.replace("+-", "- ")

adj_rsq4 = model4.rsquared_adj
print('\n\n===================================================================')
print('\r==================== Exercise 5.b.3 ===============================')

# print('\n\n(*) Here is an equation that explains relationship between returns and \nDifferentiation, Cost leadership, Efficiency and Novelty: ')
print('\n\n >> Here is an estimate for the regression model for Information\n Technology Sector: ')
print('\nReturns = ' + returns_4)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(100 * adj_rsq4))

###############################################################################
################################## 5.(b).4 ####################################
###############################################################################

print('\n\n===================================================================')
print('\r==================== Exercise 5.b.4 ===============================')

if adj_rsq2 > adj_rsq or adj_rsq3 > adj_rsq or adj_rsq3 > adj_rsq:
    print('\n\n >>  The first model explained lower variation than the splitted.')
    if adj_rsq2 > adj_rsq:
        print(
            '\n  >> The data  obtained from Financial Sector explained more variations\n in the dataset than the original model.')
    elif adj_rsq3 > adj_rsq:
        print(
            '\n >>  The data obtained from Industrial Sector explained more variations\n in the dataset than\n the original model.')
    else:
        print(
            '\n >>  The data obtained from Information Technology Sector explained more\n variations in the dataset than the original model.')
else:
    print(
        '\n\n >>  The data obtained from original model explained more variations\n in the dataset than all of the models for splitted data.')

'''

6)
a) Create a function which takes in two arguments (y) and (x), where y is the 
dependent variable and x is the explanatory variables. The function 
should run a regression and return the models' adjusted 𝑅
2
. 

'''

y = values_survey['returns']
x = values_survey[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]


def adj_rsqFunc(y, x):
    X = x
    Y = y

    # add constant
    X = sm.add_constant(X)

    # fit model
    modelF = sm.OLS(Y, X).fit()

    # obtain adjusted r-squared
    adj_rsquared = modelF.rsquared_adj

    print('\n\n===================================================================')
    print('\r==================== Exercise 6 a =================================')

    print('\n\nAdjusted r-squared = {:.5f}'.format(adj_rsquared))
    print('\n\n >> The model explains nearly {:.2f} % of the variations in returns.\n\n'.format(100 * adj_rsquared))
    print('\n\n====================================================================')
    print('\r====================================================================\n\n')


adj_rsqFunc(y=y, x=x)






