import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
# Laden der Daten
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t', encoding='latin1', na_values="n/a")
# Vorbereiten der Daten
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]
# Visualisieren der Daten
country_stats.plot(kind='scatter', x="Pro-Kopf-BIP", y='Zufriedenheit')
plt.show()
Auswahl eines linearen Modells model = sklearn.linear_model.LinearRegression()
# Trainieren des Modells
model.fit(X, y)
# Treffen einer Vorhersage für Zypern
X_new = [[22587]] # Pro-Kopf-BIP für Zypern