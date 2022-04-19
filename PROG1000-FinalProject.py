# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:20:02 2022

@author: saris
"""
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

# suppressing future warnings
warnings.filterwarnings("ignore")

###############################################################################
################################### 1.(a) #####################################
###############################################################################

# importing data set for market values
values = pd.read_csv('sp500_marketvalues.csv')

###############################################################################
################################### 1.(b) #####################################
###############################################################################

# importing data set for survey results
survey = pd.read_csv('sp500_survey.csv')

###############################################################################
################################### 1.(c) #####################################
###############################################################################

# merging the two data sets together
values_survey = values.merge(survey, on='Symbol', how='outer')

# removing the column from the dataframe as another column have similar content
values_survey.drop('Name_y', axis=1, inplace=True)

###############################################################################
################################### 1.(d) #####################################
###############################################################################

# removing empty cells from the merged data set
values_survey.dropna(inplace=True)

###############################################################################
################################### 1.(e) #####################################
###############################################################################

# list to store calculated returns values
returns_lst = []

# calculating the returns
for item1, item2 in zip(values_survey['FirmV2022'], values_survey['FirmV2017']):
    ret = ((item1 / item2) ** (1 / 5)) - 1
    returns_lst.append(ret)

# adding the calculated returns to the data set
values_survey['returns'] = returns_lst

###############################################################################
################################### 2.(a) #####################################
###############################################################################

# creating new merged dataframe
vs_2 = values.merge(survey, on='Symbol', how='outer')

# removing empty cells from newly merged dataframe
vs_2.dropna(inplace=True)

# removing the column from the dataframe as another column have similar content
vs_2.drop('Name_y', axis=1, inplace=True)

# list to store calculated returns values
vs_returns = []

# calculating the returns
for item1, item2 in zip(vs_2['FirmV2022'], vs_2['FirmV2017']):
    ret = ((item1 / item2) ** (1 / 5)) - 1
    vs_returns.append(ret)

# adding the calculated returns to the data set
vs_2['returns'] = vs_returns

# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 2.a ==========================================================')

# splitting the dataset based on sectors
vs_Financials = values_survey[values_survey['Sector'] == "Financials"]
vs_Industrials = values_survey[values_survey['Sector'] == "Industrials"]
vs_IT = values_survey[values_survey['Sector'] == "Information Technology"]

# displaying descriptive statistics for whole data set
print('\n\n >> Here are descriptive statistics for the data set: ')
print('\n >> The total number of companies sampled in the whole data set is: \n')
print(" " + str(vs_2['returns'].count()) + " Companies")
print('\n >> The different sectors where the companies sampled operates in are: \n')
print(" " + str(values_survey['Sector'].nunique()) + " Sectors\n")
print(" 1. " + values_survey['Sector'].unique()[0] + "\n 2. " + values_survey['Sector'].unique()[1] + "\n 3. " +
      values_survey['Sector'].unique()[2])
print('\n >> The average outcome for the returns based on the whole data set is: \n')
print(" " + str(round(100 * vs_2['returns'].mean(), 2)) + " %")
print(
    '\n However, no company in the data set had precisely gained the mean outcome for the returns based on this sample.\n')
print('\n >> The median outcome for returns and associated comapany info and strategy details in the data set is: \n')
vs_mn = vs_2['Name_x'][vs_2.index[vs_2['returns'] == vs_2['returns'].median()].tolist()[0]]
vs_mnr = 100 * round(vs_2['returns'][vs_2.index[vs_2['returns'] == vs_2['returns'].median()].tolist()[0]], 2)
print(" >> The company with median returns is '{}' with returns percentage of {}%. More info: \n".format(vs_mn, vs_mnr))
print(vs_2.loc[vs_2.index[vs_2['returns'] == vs_2['returns'].median()].tolist()[0]])
print('\n >> The highest value for returns and associated comapany info and strategy details in the data set is: \n')
vs_mxn = vs_2['Name_x'][vs_2.index[vs_2['returns'] == vs_2['returns'].max()].tolist()[0]]
vs_mxnr = round(100 * vs_2['returns'][vs_2.index[vs_2['returns'] == vs_2['returns'].max()].tolist()[0]], 2)
print(" >> The company with the highest returns is '{}' with returns percentage of {}%. More info: \n".format(vs_mxn,
                                                                                                              vs_mxnr))
print(vs_2.loc[vs_2.index[vs_2['returns'] == vs_2['returns'].max()].tolist()[0]])
print('\n >> The lowest value for returns and associated comapany info and strategy details in the data set is: \n')
vs_mnn = vs_2['Name_x'][vs_2.index[vs_2['returns'] == vs_2['returns'].min()].tolist()[0]]
vs_mnnr = round(100 * vs_2['returns'][vs_2.index[vs_2['returns'] == vs_2['returns'].min()].tolist()[0]], 2)
print(" >> The company with the lowest returns is '{}' with returns percentage of {}%. More info: \n".format(vs_mnn,
                                                                                                             vs_mnnr))
print(vs_2.loc[vs_2.index[vs_2['returns'] == vs_2['returns'].min()].tolist()[0]])

# displaying descriptive statistics for IT Sector
print('\n\n\n >> Here are descriptive statistics for the data set for IT Sector: ')
print('\n >> The total number of companies sampled from IT Sector in the data set is: \n')
print(" " + str(vs_IT['returns'].count()) + " Companies")
print('\n >> The average outcome for the returns in the IT Sector based on the whole data set is: \n')
print(" " + str(round(100 * vs_IT['returns'].mean(), 2)) + " %")
print(
    '\n However, no company in the IT Sector had precisely gained the mean outcome for the returns based on this sample.\n')
print('\n >> The median outcome for returns and associated comapany info and strategy details in IT Sector is: \n')
vsIT_mn = vs_IT['Name_x'][vs_IT.index[vs_IT['returns'] == vs_IT['returns'].median()].tolist()[0]]
vsIT_mnr = round(100 * vs_IT['returns'][vs_IT.index[vs_IT['returns'] == vs_IT['returns'].median()].tolist()[0]], 2)
print(" >> The company with median returns in IT Sector is '{}' with returns percentage of {}%. More info: \n".format(
    vsIT_mn, vsIT_mnr))
print(vs_IT.loc[vs_IT.index[vs_IT['returns'] == vs_IT['returns'].median()].tolist()[0]])
print('\n >> The highest value for returns and associated comapany info and strategy details in IT Sector is: \n')
vsIT_mxn = vs_IT['Name_x'][vs_IT.index[vs_IT['returns'] == vs_IT['returns'].max()].tolist()[0]]
vsIT_mxnr = round(100 * vs_IT['returns'][vs_IT.index[vs_IT['returns'] == vs_IT['returns'].max()].tolist()[0]], 2)
print(" >> The company with highest returns in IT Sector is '{}' with returns percentage of {}%. More info: \n".format(
    vsIT_mxn, vsIT_mxnr))
print(vs_IT.loc[vs_IT.index[vs_IT['returns'] == vs_IT['returns'].max()].tolist()[0]])
print('\n >> The lowest value for returns and associated comapany info and strategy details in IT Sector is: \n')
vsIT_mnn = vs_IT['Name_x'][vs_IT.index[vs_IT['returns'] == vs_IT['returns'].min()].tolist()[0]]
vsIT_mnnr = round(100 * vs_IT['returns'][vs_IT.index[vs_IT['returns'] == vs_IT['returns'].min()].tolist()[0]], 2)
print(" >> The company with lowest returns in IT Sector is '{}' with returns percentage of {}%. More info: \n".format(
    vsIT_mnn, vsIT_mnnr))
print(vs_IT.loc[vs_IT.index[vs_IT['returns'] == vs_IT['returns'].min()].tolist()[0]])

# displaying descriptive statistics for Finincial Sector
print('\n\n\n >> Here are descriptive statistics for the data set for Financial Sector: ')

print('\n >> The total number of companies sampled from Financial Sector in the data set is: \n')
print(" " + str(vs_Financials['returns'].count()) + " Companies")
print('\n >> The average outcome for the returns in the Finicial Sector based on the whole data set is: \n')
print(" " + str(round(100 * vs_Financials['returns'].mean())) + " %")
print(
    '\n However, no company in the Financial Sector had precisely gained this mean outcome for the returns based on this sample.\n')
print(
    '\n >> The median outcome for returns and associated comapany info and strategy details in the Financial Sector is: \n')
vsFin_mn = vs_Financials['Name_x'][
    vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].median()].tolist()[0]]
vsFin_mnr = round(100 * vs_Financials['returns'][
    vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].median()].tolist()[0]], 2)
print(
    " >> The company with median returns in Financial Sector is '{}' with returns percentage of {}%. More info: \n".format(
        vsFin_mn, vsFin_mnr))
print(vs_Financials.loc[vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].median()].tolist()[0]])
print(
    '\n >> The highest value for returns and associated comapany info and strategy details in Financial Sector is: \n')
vsFin_mxn = vs_Financials['Name_x'][
    vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].max()].tolist()[0]]
vsFin_mxnr = round(100 * vs_Financials['returns'][
    vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].max()].tolist()[0]], 2)
print(
    " >> The company with highest returns in Financial Sector is '{}' with returns percentage of {}%. More info: \n".format(
        vsFin_mxn, vsFin_mxnr))
print(vs_Financials.loc[vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].max()].tolist()[0]])
print('\n >> The lowest value for returns and associated comapany info and strategy details in Financial Sector is: \n')
vsFin_mnn = vs_Financials['Name_x'][
    vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].min()].tolist()[0]]
vsFin_mnnr = round(100 * vs_Financials['returns'][
    vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].min()].tolist()[0]], 2)
print(
    " >> The company with lowest returns in Financial Sector is '{}' with returns percentage of {}%. More info: \n".format(
        vsFin_mnn, vsFin_mnnr))
print(vs_Financials.loc[vs_Financials.index[vs_Financials['returns'] == vs_Financials['returns'].min()].tolist()[0]])

# displaying descriptive statistics for Industrial Sector
print('\n\n\n >> Here are descriptive statistics for the data set for Industrial Sector: ')

print('\n >> The total number of companies sampled from Industrial Sector in the data set is: \n')
print(" " + str(vs_Industrials['returns'].count()) + " Companies")
print('\n >> The average outcome for the returns in the Industrial Sector based on the whole data set is: \n')
print(" " + str(round(100 * vs_Industrials['returns'].mean(), 2)) + " %")
print(
    '\n However, no company in the Industrial Sector had precisely gained the mean outcome for the returns based on this sample.\n')
print(
    '\n >> The median outcome for returns and associated comapany info and strategy details in Industrial Sectoris: \n')
vsIndus_mn = vs_Industrials['Name_x'][
    vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].median()].tolist()[0]]
vsIndus_mnr = round(100 * vs_Industrials['returns'][
    vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].median()].tolist()[0]], 2)
print(
    " >> The company with median returns in Industrial Sector is '{}' with returns percentage of {}%. More info: \n".format(
        vsIndus_mn, vsIndus_mnr))
print(vs_Industrials.loc[
          vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].median()].tolist()[0]])
print(
    '\n >> The highest value for returns and associated comapany info and strategy details in Industrial Sector is: \n')
vsIndus_mxn = vs_Industrials['Name_x'][
    vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].max()].tolist()[0]]
vsIndus_mxnr = round(100 * vs_Industrials['returns'][
    vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].max()].tolist()[0]], 2)
print(
    " >> The company with highest returns in Industrial Sector is '{}' with returns percentage of {}%. More info: \n".format(
        vsIndus_mxn, vsIndus_mxnr))
print(
    vs_Industrials.loc[vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].max()].tolist()[0]])
print(
    '\n >> The lowest value for returns and associated comapany info and strategy details in Industrial Sector is: \n')
vsIndus_mnn = vs_Industrials['Name_x'][
    vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].min()].tolist()[0]]
vsIndus_mnnr = round(100 * vs_Industrials['returns'][
    vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].min()].tolist()[0]], 2)
print(
    " >> The company with lowest returns in Industrial Sector is '{}' with returns percentage of {}%. More info: \n".format(
        vsIndus_mnn, vsIndus_mnnr))
print(
    vs_Industrials.loc[vs_Industrials.index[vs_Industrials['returns'] == vs_Industrials['returns'].min()].tolist()[0]])

###############################################################################
################################### 2.(b) #####################################
###############################################################################

# storing dataframe's columns in variables
Y = values_survey['returns']
X1 = values_survey['Differentiation']
X2 = values_survey['Cost leadership']
X3 = values_survey['Efficiency']
X4 = values_survey['Novelty']

# finding the correlation for variables
corr_def = values_survey.corr().loc['returns', 'Differentiation']
corr_cosled = values_survey.corr().loc['returns', 'Cost leadership']
corr_effec = values_survey.corr().loc['returns', 'Efficiency']
corr_nov = values_survey.corr().loc['returns', 'Novelty']

# create figure and subplots
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

# plot subplots
ax[0, 0].scatter(X1, Y,
                 marker=".",
                 s=40,
                 color='green')
ax[0, 1].scatter(X2, Y,
                 marker=".",
                 s=40,
                 color='blue')
ax[1, 0].scatter(X3, Y,
                 marker=".",
                 s=40,
                 color='purple')
ax[1, 1].scatter(X4, Y,
                 marker=".",
                 s=40,
                 color='red')

# set y-axis labels
ax[0, 0].set_ylabel('Returns', fontsize=17)
ax[1, 0].set_ylabel('Returns', fontsize=17)
ax[0, 1].set_ylabel('Returns', fontsize=17)
ax[1, 1].set_ylabel('Returns', fontsize=17)

# set x-axis labels
ax[0, 0].set_xlabel('Differentiation', fontsize=17)
ax[0, 1].set_xlabel('Cost Leadership', fontsize=17)
ax[1, 0].set_xlabel('Efficiency', fontsize=17)
ax[1, 1].set_xlabel('Novelty', fontsize=17)

# set y-axis range
ax[0, 0].set_ylim(min(Y), max(Y))
ax[1, 0].set_ylim(min(Y), max(Y))
ax[0, 1].set_ylim(min(Y), max(Y))
ax[1, 1].set_ylim(min(Y), max(Y))

# set x-axis range
ax[0, 0].set_xlim(min(X1), max(X1))
ax[0, 1].set_xlim(min(X2), max(X2))
ax[1, 0].set_xlim(min(X3), max(X3))
ax[1, 1].set_xlim(min(X4), max(X4))

# add title to figure
fig.suptitle('Strategies and Returns', fontsize=30)

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
Fname2b = '2b_scatterVS.png'
fig.savefig(Fname2b,  # name (and path) for storing image
            dpi=300,  # image resolution
            bbox_inches='tight')  # remove white space around image

print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 2 b ==========================================================')
print("\n\n >> A figure with the name '{}' has been created and saved in the device.\n\n".format(Fname2b))

###############################################################################
################################### 3.(a) #####################################
###############################################################################

# sector names

x_sec = sorted(values_survey['Sector'].unique().tolist(), key=len)

# returns in percentage for different sectors
y_sec = round(100 * values_survey.groupby(['Sector'])['returns'].mean(), 2)

# create figure and subplots
fig, ax = plt.subplots()

# plot subplots
ax.bar(x_sec, y_sec, 0.5,
       facecolor='lightblue',
       edgecolor='blue',
       alpha=1,
       hatch='/')

# set axis label
ax.set_ylabel('Returns', color='blue', fontsize=16)
ax.set_xlabel('Sectors', color='blue', fontsize=16)

# set figure title
ax.set_title("Average sectors' returns", fontsize=24)

# hide the right and top spines
ax.spines[['right', 'top']].set_visible(False)

# save figure
Fname3a = '3a_barSectors.png'
plt.savefig(Fname3a,  # name (and path) for storing image
            dpi=300,  # image resolution
            bbox_inches='tight')  # remove white space around image

print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 3 a ==========================================================')
print("\n\n >> A figure with the name '{}' has been created and saved in the device.\n\n".format(Fname3a))

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

# set axis labels
ax[0].set_xlabel('Returns', color='green', fontsize=16)
ax[0].set_ylabel('Number of Companies', color='green', fontsize=16)

ax[1].set_xlabel('Returns', color='blue', fontsize=16)
ax[1].set_ylabel('Number of Companies', color='blue', fontsize=16)

ax[2].set_xlabel('Returns', color='purple', fontsize=16)
ax[2].set_ylabel('Number of Companies', color='purple', fontsize=16)

# set figure title
fig.suptitle("Sectors' Returns", fontsize=27)

# add subplot title
ax[0].set_title('Industrial Sector', fontsize=17)
ax[1].set_title('Information Technology Sector', fontsize=17)
ax[2].set_title('Financial Sector', fontsize=17)

# hide the right and top spines
ax[0].spines[['right', 'top']].set_visible(False)
ax[1].spines[['right', 'top']].set_visible(False)
ax[2].spines[['right', 'top']].set_visible(False)

# save figure
Fname3b = '3b_histSectors.png'
plt.savefig(Fname3b,  # name (and path) for storing image
            dpi=300,  # image resolution
            bbox_inches='tight')  # remove white space around image

print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 3 b ==========================================================')
print("\n\n >> A figure with the name '{}' has been created and saved in the device.\n\n".format(Fname3b))

###############################################################################
################################### 4.(a) #####################################
###############################################################################

# assigning column(s) to variables
X = values_survey[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]
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

# model regressing equation
returns = str(alpha) + " +" + str(beta1) + " Differentiation  +" + str(beta2) + " Cost \
leadership +" + str(beta3) + " Efficiency +" + str(beta4) + " Novelty"

# modifying regression equation
if "+-" in returns:
    returns = returns.replace("+-", "- ")
if " +" in returns:
    returns = returns.replace(" +", " + ")

# obtaining adjusted r-squared for the model
adj_rsq = model.rsquared_adj
adj_pros = round(100 * adj_rsq, 2)

# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 4 a ==========================================================')

# dispalying exercise solution
print('\n\n >> Here is an estimate for the regression model: ')
print('\nReturns = ' + returns)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(adj_pros))

###############################################################################
################################### 4.(b) #####################################
###############################################################################'

# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r====================================================== Exercise 4 b ======================================================')

# values given in exercise
# removing hashtags enables dynamic insertion of values
Differentiation = 54  # int(input('Differentiation: '))
Cost_leadership = 61  # int(input('Cost leadership: '))
Efficiency = 57  # int(input('Efficiency: '))
Novelty = 42  # int(input('Novelty: '))

# calculating the returns
returns = alpha + beta1 * Differentiation + beta2 * Cost_leadership + beta3 * Efficiency \
          + beta4 * Novelty
returns_pros = round(100 * returns, 2)

# dispalying exercise solution
print('\n\nReturns = {:.5f} (i.e. almost {} %)'.format(returns, returns_pros))

###############################################################################
################################### 4.(c) #####################################
###############################################################################

# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r====================================================== Exercise 4 c ======================================================')

# values given in exercise
# removing hashtags enables dynamic insertion of values
Differentiation = 54  # int(input('Differentiation: '))
Cost_leadership = 61  # int(input('Cost leadership: '))
Efficiency = 57  # int(input('Efficiency: '))
Novelty = 55  # int(input('Novelty: '))

# calculating the returns
returns = alpha + beta1 * Differentiation + beta2 * Cost_leadership + beta3 * Efficiency \
          + beta4 * Novelty
returns_pros = round(100 * returns, 2)

# dispalying exercise solution
print('\n\nReturns = {:.5f} (i.e. almost {} %)'.format(returns, returns_pros))

###############################################################################
################################### Extra 10 ##################################
###############################################################################
"""
print('\n\n==========================================================================================================================')
print('\r=============================================== Extra 10 =================================================================')


#values given in exercise
#removing hashtags enables dynamic insertion of values
Differentiation= float(input('Differentiation: '))
Cost_leadership= float(input('Cost leadership: '))
Efficiency= float(input('Efficiency: '))
Novelty= float(input('Novelty: '))

#calculating the returns
returns = alpha+beta1*Differentiation+beta2*Cost_leadership+beta3*Efficiency\
    +beta4*Novelty
returns_pros = round(100*returns,2)

#dispalying exercise solution
print('\n\nReturns = {:.5f} (i.e. almost {} %)'.format(returns, returns_pros))
"""
###############################################################################
################################### 5.(a) #####################################
###############################################################################

# splitting the dataset based on sectors
vs_Financials = values_survey[values_survey['Sector'] == "Financials"]
vs_Industrials = values_survey[values_survey['Sector'] == "Industrials"]
vs_IT = values_survey[values_survey['Sector'] == "Information Technology"]

###############################################################################
################################## 5.(b).1 ####################################
###############################################################################

# assigning column(s) to variables
X_2 = vs_Financials[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]
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

# model regressing equation
returns_2 = str(alpha2) + " +" + str(beta1_2) + " Differentiation  +" + str(beta2_2) + " Cost \
leadership +" + str(beta3_2) + " Efficiency +" + str(beta4_2) + " Novelty"

# modifying regression equation
if "+-" in returns_2:
    returns_2 = returns_2.replace("+-", "- ")
if " +" in returns_2:
    returns_2 = returns_2.replace(" +", " + ")

# obtaining adjusted r-squared for the model
adj_rsq2 = model2.rsquared_adj
adj_pros2 = 100 * adj_rsq2
# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 5.b.1 ========================================================')

# dispalying exercise solution
print('\n\n >> Here is an estimate for the regression model for Financial Sector: ')
print('\nReturns = ' + returns_2)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(adj_pros2))

###############################################################################
################################## 5.(b).2 ####################################
###############################################################################

# assigning column(s) to variables
X_3 = vs_Industrials[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]
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

# model regressing equation
returns_3 = str(alpha3) + " +" + str(beta1_3) + " Differentiation  +" + str(beta2_3) + " Cost \
leadership +" + str(beta3_3) + " Efficiency +" + str(beta4_3) + " Novelty"

# modifying regression equation
if "+-" in returns_3:
    returns_3 = returns_3.replace("+-", "- ")
if " +" in returns_3:
    returns_3 = returns_3.replace(" +", " + ")

# obtaining adjusted r-squared for the model
adj_rsq3 = model3.rsquared_adj
adj_pros3 = 100 * adj_rsq3

# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 5.b.2 ========================================================')

# dispalying exercise solution
print('\n\n >> Here is an estimate for the regression model for Industrial Sector: ')
print('\nReturns = ' + returns_3)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(adj_pros3))

###############################################################################
################################## 5.(b).3 ####################################
###############################################################################

# assigning column(s) to variables
X_4 = vs_IT[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]
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

# model regressing equation
returns_4 = str(alpha4) + " +" + str(beta1_4) + " Differentiation  +" + str(beta2_4) + " Cost \
leadership +" + str(beta3_4) + " Efficiency +" + str(beta4_4) + " Novelty"

# modifying regression equation
if "+-" in returns_4:
    returns_4 = returns_4.replace("+-", "- ")
if " +" in returns_4:
    returns_4 = returns_4.replace(" +", " + ")

# obtaining adjusted r-squared for the model
adj_rsq4 = model4.rsquared_adj
adj_pros4 = 100 * adj_rsq4

# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 5.b.3 ========================================================')

# dispalying exercise solution
print('\n\n >> Here is an estimate for the regression model for Information Technology Sector: ')
print('\nReturns = ' + returns_4)
print('\n\n >> The model explains {:.2f} % of the variations in returns.\n\n'.format(adj_pros4))

###############################################################################
################################## 5.(b).4 ####################################
###############################################################################

# displaying exercise number
print(
    '\n\n==========================================================================================================================')
print(
    '\r================================================== Exercise 5.b.4 ========================================================')

print('\n\n >>  Explanatory power for the model from all three sectors: {} %'.format(round(adj_pros)))
print('\n >>  Explanatory power for the model from IT Sector: {} %'.format(round(adj_pros4)))
print('\n >>  Explanatory power for the model from Financial Sector: {} %'.format(round(adj_pros2)))
print('\n >>  Explanatory power for the model from Industrial Sector: {} %'.format(round(adj_pros3)))

# checking the models to find which one explain more variations
if adj_rsq2 > adj_rsq or adj_rsq3 > adj_rsq or adj_rsq3 > adj_rsq:
    print('\n\n >>  The first model explained lower variation than the splitted.')
    if adj_rsq2 > adj_rsq:
        print(
            '\n  >> The data  obtained from Financial Sector explained more variations in the dataset than the original data set for three sectors.')
    elif adj_rsq3 > adj_rsq:
        print(
            '\n >>  The data obtained from Industrial Sector explained more variations in the dataset than\n the original data set for three sectors.')
    else:
        print(
            '\n >>  The data obtained from Information Technology Sector explained more variations in the dataset than the original data set for three sectors.')
else:
    print('\n\n >>  The data obtained from original data set explained more variations than all three splitted data sets.')

###############################################################################
################################## 6.(a) ######################################
###############################################################################

# assigning column(s) to variables
y = values_survey['returns']
x = values_survey[['Differentiation', 'Cost leadership', 'Efficiency', 'Novelty']]


# creating a function
def adj_rsqFunc(y, x):
    X = x
    Y = y

    # add constant
    X = sm.add_constant(X)

    # fit model
    modelF = sm.OLS(Y, X).fit()

    # obtain adjusted r-squared
    adj_rsquared = modelF.rsquared_adj
    adj_pros = 100 * adj_rsquared

    # displaying exercise number
    print(
        '\n\n==========================================================================================================================')
    print(
        '\r================================================== Exercise 6 a ==========================================================')

    # dispalying exercise solution
    print('\n\n >> Adjusted r-squared = {:.5f}'.format(adj_rsquared))
    print('\n\n >> The model explains nearly {:.2f} % of the variations in the model.\n\n'.format(adj_pros))
    print(
        '\n\n===========================================================================================================================')
    print(
        '\r===========================================================================================================================\n\n')


adj_rsqFunc(y=y, x=x)
