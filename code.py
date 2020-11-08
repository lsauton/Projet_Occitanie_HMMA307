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
path_target = "./datasets/Mesure_journaliere.csv"
occitanie =  pd.read_csv(path_target, sep="," , header=0)

# Nous allons faire notre étude sur 3 périodes: sur une année, pendant l'été 2020 et pendant le confinement. 
year = [ '2019-11','2019-12','2020-01','2020-02',   '2020-03', '2020-04',
        '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11']
ete = year[7:10]
confinement = year[4:7]

# Nous allons selectionner 3 villes proches géographiquement et 3 villes éloignées
sparsed_city = 'MONTPELLIER','TOULOUSE', 'PERPIGNAN'
closed_city =  'MONTPELLIER','SAINT-GELY-DU-FESC', 'LATTES'





################################################################
#           Creation des fonctions utile pour le projet 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Fonction get_data nous permet d extraire de notre base l'es données nécéssaires à notre études
def get_data(occ_df, polluant ='O3', period = 'M', villes=['MONTPELLIER'], date=['2019-12'] ):
    
    occ_df['date'] = pd.to_datetime(occ_df['date_debut']).dt.to_period(period)
    occ_df = occ_df.sort_values(by = 'date', ascending = True)
    occ_df = occ_df[occ_df['nom_poll'] == polluant] 
    occ_df['standard'] = (occ_df[['valeur']] - np.mean(occ_df[['valeur']]))/ np.std(occ_df[['valeur']])
    variables = ['X', 'Y', 'nom_com', 'nom_station', 'valeur', 'date', 'standard','typologie','influence']
    occ_df = occ_df[variables]
    occ_df = occ_df[occ_df['nom_com'].isin(villes)]
    occ_df_date = occ_df[occ_df['date'] == date[0]]
    
    for i in date[1:]:
        dat = occ_df[occ_df['date'] == i] 
        occ_df_date = pd.concat([occ_df_date, dat])

    
    return occ_df_date


def get_mean(occ_df, villes, poll ):
     
    occ_df = occ_df[occ_df['nom_com'] == villes] # selection of Montpellier
    occ_df = occ_df[occ_df['nom_poll'] == poll ] # Only Prés d' Arènes urbain available for O3 
    occ_df['date'] = pd.to_datetime(occ_df['date_debut']).dt.to_period('M') #.dt.to_period('M') good format for the date
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



#NO2
mtp_NO2 = get_mean(occitanie, 'MONTPELLIER', 'NO2')
tlse_NO2 = get_mean(occitanie, 'TOULOUSE', 'NO2')
perpi_NO2 = get_mean(occitanie, 'PERPIGNAN', 'NO2')



plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
ax = plt.gca()
mtp_O3.plot(kind='line', x='date', y='valeur', ax=ax, color='r', label='Montpellier')
tlse_O3.plot(kind='line', x='date', y='valeur', color='b', ax=ax, label="Toulouse")
perpi_O3.plot(kind='line', x='date', y='valeur', color='g', ax=ax, label="Perpignan")
plt.title("Evolution de l'O3 par mois de Novembre 2019 and Novembre 2020 ")

plt.subplot(1, 2, 2)
ax = plt.gca()
mtp_NO2.plot(kind='line', x='date', y='valeur', ax=ax, color='r', label='Montpellier')
tlse_NO2.plot(kind='line', x='date', y='valeur', color='b', ax=ax, label="Toulouse")
perpi_NO2.plot(kind='line', x='date', y='valeur', color='g', ax=ax, label="Perpignan")
plt.title("Evolution du NO2 par mois de Novembre 2019 and Novembre 2020 ")

plt.savefig(os.path.join("./Images", "Evolution_O3_NO2.pdf"))
plt.show()



plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
ax = plt.gca()
mtp_O3.plot(kind='line', x='date', y='valeur', ax=ax, color='r', label='Montpellier')
tlse_O3.plot(kind='line', x='date', y='valeur', color='b', ax=ax, label="Toulouse")
perpi_O3.plot(kind='line', x='date', y='valeur', color='g', ax=ax, label="Perpignan")
plt.title("Evolution de l'O3 par mois de Novembre 2019 and Novembre 2020 ")

plt.subplot(1, 2, 2)
ax = plt.gca()
mtp_NO2.plot(kind='line', x='date', y='valeur', ax=ax, color='r', label='Montpellier')
tlse_NO2.plot(kind='line', x='date', y='valeur', color='b', ax=ax, label="Toulouse")
perpi_NO2.plot(kind='line', x='date', y='valeur', color='g', ax=ax, label="Perpignan")
plt.title("Evolution du NO2 par mois de Novembre 2019 and Novembre 2020 ")

plt.savefig(os.path.join("C:/Users/Lolo/Documents/M2/Github/Projet Pollution 307/Images", "Evolution_O3_NO2.pdf"))
plt.show()



#######################################################################################################################################

####################################################################
#           ANOVA test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Villes d'interet pour l'etude

# On sotck les data preparer par la fonction get data



spars_O3_conf = get_data(occitanie, villes= sparsed_city, date= confinement )
closed_O3_conf = get_data(occitanie, villes=closed_city, date= confinement )

spars_O3_y = get_data(occitanie, villes= sparsed_city, date= year )
closed_O3_y = get_data(occitanie, villes=closed_city, date= year )






# Pour le confinement
fit_sparsed_c = ols('valeur ~ C(nom_com )', data= spars_O3_conf).fit()
fit_closed_c = ols('valeur ~ C(nom_com )', data= closed_O3_conf).fit()
anova_O3_s = sm.stats.anova_lm(fit_sparsed_c, typ=2) 
anova_O3_cl = sm.stats.anova_lm(fit_closed_c, typ=2) 
print(anova_O3_s) 
print(anova_O3_cl) 

# pour l'année 2020
fit_sparsed_y = ols('valeur ~ C(nom_com )', data= spars_O3_y).fit()
fit_closed_y = ols('valeur ~ C(nom_com )', data= closed_O3_y).fit()
anova_O3_sy = sm.stats.anova_lm(fit_sparsed_y, typ=2) 
anova_O3_cly = sm.stats.anova_lm(fit_closed_y, typ=2) 
print(anova_O3_sy) 
print(anova_O3_cly) 



fig, ax = plt.subplots()
_, (__, ___, r) = sp.stats.probplot(fit_sparsed_c.resid, plot=ax, fit=True)
plt.savefig(os.path.join("C:/Users/Lolo/Documents/M2/Github/Projet Pollution 307/Images", "Verification_des_résidues.pdf"))
plt.show()