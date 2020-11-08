# importation des packages necessaires pour l ealisation du projet 

import numpy as np
import pandas as pd
import statsmodels.api as sm
from download import download 
from statsmodels.formula.api import ols
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt 
import os


# chargement des données 
path_target = "C:/Users/Lolo/Documents/M2/Github/Projet Pollution 307/Mesure_journaliere.csv"
occitanie =  pd.read_csv(path_target, sep="," , header=0)


################################################################
#           Creation des fonctions utile pour le projet 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Fonction get_data nous permet d extraire de notre base l'es données nécéssaires à notre études
def get_data(occ_df, polluant ='O3', period = 'M', villes=['MONTPELLIER'] ):
    
    occ_df['date'] = pd.to_datetime(occ_df['date_debut']).dt.to_period(period)
    occ_df = occ_df[occ_df['nom_poll'] == polluant] 
    occ_df['standard'] = (occ_df[['valeur']] - np.mean(occ_df[['valeur']]))/ np.std(occ_df[['valeur']])
    variables = ['X', 'Y', 'nom_com', 'nom_station', 'valeur', 'date', 'standard']
    occ_df = occ_df[variables]
    occ_df = occ_df[occ_df['nom_com'].isin(villes)]
    occitanie.sort_values(by = 'date', ascending = True)
    return occ_df



def get_mean(occ_df, villes, poll ):
     
    occ_df = occ_df[occ_df['nom_com'] == villes] # selection of Montpellier
    occ_df = occ_df[occ_df['nom_poll'] == poll ] # Only Prés d' Arènes urbain available for O3 
    occ_df['date'] = pd.to_datetime(occ_df['date_debut']).dt.to_period('M') # good format for the date
    occ_df = occ_df.sort_values(by = 'date', ascending = True) # date sorting
    city_df = occ_df.groupby('date').agg({'valeur':'mean'}) # mean to have value for months
    city_df['date'] = occ_df.date.unique()
    city_df['nom_com'] = [villes]*13
    city_df['confinement'] = ['no'] * 13
    city_df['confinement'][4:7] = city_df['confinement'][12] = 'yes'
    return city_df







################################################################
#           Plotting a mean of O3 pollution for 3 cities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## O3

mtp_O3 = get_mean(occitanie, 'MONTPELLIER', 'O3')
tlse_O3 = get_mean(occitanie, 'TOULOUSE', 'O3')
perpi_O3 = get_mean(occitanie, 'PERPIGNAN', 'O3')



plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
ax = plt.gca()
mtp_O3.plot(kind='line', x='date', y='valeur', ax=ax, color='r', label='Montpellier')
tlse_O3.plot(kind='line', x='date', y='valeur', color='b', ax=ax, label="Toulouse")
perpi_O3.plot(kind='line', x='date', y='valeur', color='g', ax=ax, label="Carcassonne")
plt.title("O3 evolution over months between November 2019 and November 2020")
plt.show()

#NO2
mtp_NO2 = get_mean(occitanie, 'MONTPELLIER', 'NO2')
tlse_NO2 = get_mean(occitanie, 'TOULOUSE', 'NO2')
perpi_NO2 = get_mean(occitanie, 'PERPIGNAN', 'NO2')



plt.figure(figsize=(15,5))
plt.subplot(1, 2, 2)
ax = plt.gca()
mtp_NO2.plot(kind='line', x='date', y='valeur', ax=ax, color='r', label='Montpellier')
tlse_NO2.plot(kind='line', x='date', y='valeur', color='b', ax=ax, label="Toulouse")
perpi_NO2.plot(kind='line', x='date', y='valeur', color='g', ax=ax, label="Perpignan")
plt.title("NO2 evolution over months between November 2019 and November 2020")
plt.show()



####################################################################
#           ANOVA test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sparsed_city = 'MONTPELLIER','TOULOUSE', 'PERPIGNAN'
closed_city =  'MONTPELLIER','NIMES', 'LATTES'

spars_O3 = get_data(occitanie, villes= sparsed_city)
closed_O3 = get_data(occitanie, villes=closed_city)

sns.countplot(spars_O3['nom_com'])
plt.show()

sns.countplot(closed_O3['nom_com'])
plt.show()

fit_sparsed = ols('standard ~ C(nom_com )', data= spars_O3).fit()
residu = fit_sparsed.resid 
aov_tableO3 = sm.stats.anova_lm(fit_sparsed)
aov_tableO3

fit_closed = ols('standard ~ C(nom_com )', data= closed_O3).fit()
residu = fit_closed.resid 
aov_tableO3_c = sm.stats.anova_lm(fit_closed)
aov_tableO3_c

sns.violinplot(data=closed_O3, x='nom_com', y='valeur', fit= True)
plt.xlabel('Villes')
plt.ylabel('O3') 
plt.title("violinplot de l'Ozone par les villes ")
plt.show()