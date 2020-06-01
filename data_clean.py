# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:40:33 2020

@author: Guillermo Camps Pons
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from scipy.stats import shapiro, kstest, fligner, ttest_ind, mannwhitneyu
from scipy.stats import pearsonr, spearmanr

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 100)
plt.close("all")
out_folder = 'output/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    print('hi')

def describe_df(df, name=''):
    '''Descripció formatejada d'un dataframe (df) amb nom (name)'''
    
    print('\nColumnes del DataFrame {:s}:'.format(name))
    
    print('\t{:<16s}{:>16s}{:>16s}{:>16s}'.format('Atribut',
                                                  'Tipus',
                                                  'Nombre de nuls', 
                                                  'Valors unics'))
    print('\t'+'-'*16*4)
    for col, typ, nul in zip(df.columns, df.dtypes, df.isna().sum()):        
        print('\t{:<16s}{:>16s}{:>16d}{:>16d}'.format(col, 
                                                      str(typ),
                                                      nul, 
                                                      len(df[col].unique())))
    print('Nombre de files: {:d}'.format(df.shape[0]))
    print('Nombre de columnes: {:d}\n'.format(df.shape[1]))
    return

def create_heatmap(mat, vmin, vmax, cmap = 'coolwarm',
                    num = None, figsize = (8,8), out_folder='output/'):
    """Crea un heatmap a partir d'una matriu en format pd.DataFrame"""
    fig, ax = plt.subplots(1,1,figsize=figsize, num = num)
    plot = ax.matshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, '{:.4f}'.format(mat.iloc[i,j]), 
                    ha="center", va="center")
    ax.set(xticks=range(len(mat.columns)), xticklabels = mat.columns,
           yticks=range(len(mat.index)),   yticklabels = mat.index,
           xlabel = mat.columns.name, ylabel = mat.index.name, 
           title = num+'\n')
    ax.xaxis.set_label_position('top')
    fig.colorbar(plot)
    plt.tight_layout()
    plt.savefig(out_folder+'fig_{:s}.png'.format(num))
    plt.show()
    return

def sep_(title):
    l = 75-len(title)
    print('\n'+'-'*int(np.floor(l/2))+title+'-'*int(np.ceil(l/2))+'\n')
    return

# =============================================================================
# ~ Càrrega d'arxius
# =============================================================================
sep_("Càrrega d'arxius")

# Arxiu train.csv
train = pd.read_csv('data/train.csv')
describe_df(train,'train')

# Arxiu test.csv
# Veiem que, en comparació a train.csv, no hi ha la variable Survived.
test = pd.read_csv('data/test.csv')
describe_df(test,'test')

# Arxiu gender_submission.csv
# Aquest arxiu conté prediccions de la variable Survived del conjunt de test 
# on s'assumeix que les dones sobreviveixen. No l'utilitzarem.
survived = pd.read_csv('data/gender_submission.csv')
describe_df(survived,'survived')
del survived
    
# =============================================================================
# ~ Integració de les dades
# =============================================================================
sep_("Integració")

# Creem una variable per a identificar si els registres provenen de l'arxiu
# test o no
train.loc[:,'Test'] = 0
test.loc[:,'Test']  = 1

# Per facilitar el tractament de les dades, separem Survived del conjunt de 
# train i ajuntem totes les dades en un DataFrame.
y = train.Survived
data = train.drop('Survived', axis = 1).append(test,ignore_index=True)
describe_df(data, 'sencer')

# =============================================================================
# ~ Exploració de les dades
# =============================================================================
sep_("Exploració")

# Veiem que el nombre de valors únics de Name ara no és igual al nombre de 
# files, quan pels datasets separats sí ho era. Això pot significar que hi 
# files duplicades hagi. Veiem que no són exactament iguals.
noms_dup = data.Name[data.Name.duplicated()]
print('\nFiles amb Name duplicat')
print(data.loc[data.Name.isin(noms_dup)].sort_values('Name'))

# Mostrem les primeres 5 files per a veure el contingut dels atributs.
# Veiem que les columnes de tipus object corresponen a cadenes de caràcters.
# Llavors no hi ha cap tipus de numèric que hagi estat mal interpretat o que 
# tingui un mal format.
print('\nPrimeres files')
print(data.head(5))

# ---- Distribucions ----------------------------------------------------------
num = 'Distrib'
fig, axes = plt.subplots(2,4, figsize=(20,10), num = num)
# Numèrics -> histograma
for var, ax in zip(['Age','Fare'], axes[0]):
    d = data[var].dropna()
    ax.hist(d, bins=20)
    ax.set_title('Distribució de {:s}'.format(var))
# Enters -> histograma amb pas=1 (equivalent a barres)
for var, ax in zip(['SibSp','Parch'], axes[0][2:]):
    d = data[var].dropna()
    ax.hist(data[var], bins = np.arange(-0.5,10.5,1))
    ax.set(xticks = np.arange(0,11,1), 
           title='Distribució de {:s}'.format(var))
# Ordinals i categòrics -> barres
for var, ax in zip(['Pclass','Sex','Embarked'], axes[1]):
    d = data[var].value_counts().sort_index()
    ax.bar(range(len(d)), d)
    ax.set(xticks=range(len(d)), 
           xticklabels = d.index, 
           title = 'Distribució de {:s}'.format(var))
axes[1][-1].set_visible(False)
plt.tight_layout()
plt.savefig(out_folder+'fig_{:s}.png'.format(num))
plt.show()

# =============================================================================
# ~ Tractament de variables
#       - Síntesi de variables
#       - Tractament de nuls
#       - Tractament de valors extrems
# =============================================================================
sep_("Tractament de variables")

# Per l'anàlisi la variable Ticket no ens pot aportar valor ja que són valors 
# arbitraris. PassangerId tampoc ens serveix per l'anàlisi però identifica els 
# passatgers.
data.drop(['Ticket'], axis = 1, inplace=True)

# ---- Name -------------------------------------------------------------------
#          - Sintesi -
# De la variable Name podem extrure els títols com Mr. o Mrs. Veiem que hi ha 
# alguns títols poc comuns i els substituim per 'Other'. Després creem 
# variables binàries a partir de la resta.
data.loc[:,'Title']= data.Name.str.extract('(, .+\.)')[0].str.replace(', |\.',
                                                                      '')
print(data.Title.value_counts())
data.loc[-data.Title.isin(['Mr','Miss','Mrs','Master']), 'Title'] ='Other'
data = data.join(pd.get_dummies(data.Title,prefix = 'Title')
                 .drop('Title_Other',axis=1)) 
data.drop(['Name','Title'], axis = 1, inplace=True)

# ---- Cabin ------------------------------------------------------------------
#          - Sintesi -
# La variable cabin té molts valors nuls així que creem una nova variable 
# segons si té un valor o no.
data.loc[:,'Cabin_any'] = (-data.Cabin.isna()).astype('uint8')
data.drop('Cabin', axis = 1, inplace=True)

# ---- Sex --------------------------------------------------------------------
#          - Sintesi -
# Creem una variables binària a partir de Sex (male). Descartem una de 
# les dues perquè female = -male.
data = data.join(pd.get_dummies(data.Sex, 
                                drop_first=True, 
                                prefix = 'Sex'))    
data.drop('Sex', axis = 1, inplace=True)

# ---- Embarked ---------------------------------------------------------------
#          - Nuls -
# Aquesta variable és categòrica i només té dos valors nuls. Com són pocs 
# valors podem substituir el valor pel més frequent (S) amb SimpleImputer, ja 
# que, com hem vist a les distribucions, és més frequent que la resta.
var = 'Embarked'
nuls = data[var].isna()
data.loc[:,var] = SimpleImputer(strategy='most_frequent').fit_transform(
    data.loc[:,var].values.reshape(-1, 1))
print('\nImputació de valors nuls de {:s}:'
      .format(var),data.loc[nuls], sep='\n')

#          - Sintesi -
# Creem dues variables binàries a partir d'Embarked (Q, S). Descartem una de 
# les tres perquè C = -(Q & S).
data = data.join(pd.get_dummies(data.Embarked, 
                                drop_first=True, 
                                prefix = 'Embarked'))
data.drop('Embarked', axis = 1, inplace=True)
  
# ---- Fare -------------------------------------------------------------------
#          - Nuls -
# Aquesta variable és numèrica i només té un valor nul. Com són pocs valors, 
# podem imputar el valor amb un KNNImputer amb la variable Pclass, ja que, 
# com veurem a l'apartat de correlacions, té una mica de correlació.
var = 'Fare'
nuls = data[var].isna()
data[['Pclass',var]] = KNNImputer(
    n_neighbors = 20).fit_transform(data[['Pclass',var]])
print('\nImputació de valors nuls de {:s}:'
      .format(var),data.loc[nuls],sep='\n')

#          - Extrems -
# Els valors amb Fare~500 són molt llunyans. Els substituim pel 2n màxim
num ='Box_Fare'
fig, ax = plt.subplots(1,1,figsize=(4,8), num = num)
data.boxplot('Fare',ax=ax,grid=False)
plt.savefig(out_folder+'fig_{:s}.png'.format(num))
plt.show()

data.loc[data.Fare==data.Fare.max(),'Fare'] = pd.NA
nuls = data[var].isna()
data.loc[:,var] = data.loc[:,var].fillna(data.Fare.max())
print('\nImputació de valors extrems de {:s}:'
      .format(var),data.loc[nuls], sep='\n')

# ---- Age --------------------------------------------------------------------
#          - Nuls -
# La variable Age conté nu nombre elevat de valors nuls. Provem d'imputar amb
# un KNN amb totes les variables (menys Test i PassengerId).
var = 'Age'
X = data.copy().drop(['PassengerId','Test'], axis=1)
nuls = data[var].isna()
X.loc[:,:] = KNNImputer(n_neighbors = 20).fit_transform(X)
print('\nImputació de valors nuls de {:s}:'
      .format(var),X.loc[nuls],sep='\n')

# Comparem les distribucions
num = 'Imputacio Age'
fig, ax = plt.subplots(1,1,figsize=(8,8), num = num)
ax.hist([data[var], X[var]], bins=20, label=['Original','Nuls imputats'])
ax.set_title('Variable age')
ax.legend()
plt.savefig(out_folder+'fig_{:s}.png'.format(num))
plt.show()

# El pic de la distribució clarament ha canviat però s'han distribuit els punts
# al llarg de diferents anys. Acceptem aquesta imputació.
data = X.copy().join(data[['PassengerId','Test']])

#          - Extrems -
# Els valors de l'edat són raonables així que els deixem.
num ='Box_Age'
fig, ax = plt.subplots(1,1,figsize=(4,8), num = num)
data.boxplot('Age',ax=ax,grid=False)
plt.savefig(out_folder+'fig_{:s}.png'.format(num))
plt.show()

# =============================================================================
# ~ Output neteja de dades
# =============================================================================
sep_("Arxius netejats")

# Corregim alguns formats que han canviat amb impute
int_cols = data.columns[~data.columns.isin(['Age','Fare'])]
data= data.astype({x:'int32' for x in int_cols})

# Recuperem els arxius
train = data.loc[data.Test==0].drop('Test', axis = 1).join(y)
train.to_csv(out_folder+'train_clean.csv', index=False)

test  = data.loc[data.Test==1].drop('Test', axis = 1)
test.to_csv( out_folder+'test_clean.csv',  index=False)

# =============================================================================
# ~ Anàlisi - Inicial
# =============================================================================
sep_("Anàlisi Incial")

# Llevem la variable PassengerId
X = train.drop('PassengerId', axis=1)         
X0 = X.loc[y==0,:]
X1 = X.loc[y==1,:]

describe_df(X, "d'analisi")

# ---- Estadística descriptiva ------------------------------------------------
print('Descripció atributs categòrics i enters')
print(X.astype('object').describe().drop(['Age','Fare'],axis=1))
print('\nDescripció atributs numèrics')
print(X.describe())


# =============================================================================
# ~ Anàlisi - Normalitat
# =============================================================================
sep_("Anàlisi Normalitat")

# Agafem un nivell de significàcia alpha = 0.05 per a avaluar la normalitat 
# sota el test de Shapiro-Wilk. Obtenim que clarament ninguna de les variables 
# numèriques segueix una distribució normal.
print('\n--Test Shapiro-Wilk:', 
      '\tH0: les dades segueixen una distribució normal',
      '\tH1: les dades no segueixen una distribució normal', sep = '\n')
alpha = 0.05
res = pd.DataFrame(columns = ['Variable','Survived',
                                      'Estadistic','p-valor','H0'])
for var in ['Age','Fare','SibSp','Parch']:
    for i,d in enumerate([X0,X1]):
        stat, p = shapiro(d[var])
        res = res.append({'Variable': var,
                          'Survived': i,
                          'Estadistic': stat,
                          'p-valor': round(p,6),
                          'H0': p>=alpha},
                         ignore_index = True)
print(res)     

# Agafem un nivell de significàcia alpha = 0.05 per a avaluar la normalitat 
# sota el test de Kolmogorov-Smirnov. Obtenim que clarament ninguna de les  
# variables numèriques segueix una distribució normal.
print('\n--Test Kolmogorov-Smirnov:', 
      '\tH0: les dades segueixen una distribució normal',
      '\tH1: les dades no segueixen una distribució normal', sep = '\n')
alpha = 0.05
res = pd.DataFrame(columns = ['Variable','Survived',
                                      'Estadistic','p-valor','H0'])
for var in ['Age','Fare','SibSp','Parch']:
    for i,d in enumerate([X0,X1]):
        stat, p = kstest(d[var], 'norm')
        res = res.append({'Variable': var,
                          'Survived': i,
                          'Estadistic': stat,
                          'p-valor': round(p,6),
                          'H0': p>=alpha},
                         ignore_index = True)
print(res)     

# =============================================================================
# ~ Anàlisi - Homoscedasticitat
# =============================================================================    
sep_("Anàlisi Homoscedasticitat")

# Agafem un nivell de significàcia de 0.05 per a avaluar la homoscedasticitat 
# entre els dos grups sota el test de Fligner-Killeen, donat que no hi ha 
# normalitat. Obtenim que l'única variable amb homoscedasticitat és SibSp.
print('\n--Test Fligner-Killeen:', 
      '\tH0: la variància és igual en ambdós grups (homoscedasticitat)',
      '\tH1: la variància no és igual entre els grups (heteroscedasticitat)', 
      sep = '\n')
alpha = 0.05
res = pd.DataFrame(columns = ['Variable','Estadistic','p-valor','H0'])
for var in ['Age','Fare','SibSp','Parch']:
    stat, p = fligner(X0[var], X1[var])
    res = res.append({'Variable': var,
                      'Estadistic': stat,
                      'p-valor': round(p,6),
                      'H0': p>=alpha}, ignore_index = True)
print(res)
res_homo = res.copy()

# =============================================================================
# ~ Anàlisi - Tendència central
# =============================================================================
sep_("Anàlisi Tendència Central")

# ---- t-Student test ---------------------------------------------------------
# Encara que no podem assumir la normalitat de les mostres, com que tenim unes 
# mostres prou grans (N>30), la mitjana de les mostres segueix una distribució
# normal. Per a avaluar si les mitjanes són iguals podem fer un test de 
# t-Student. El cas de variàncies diferents també es coneix com test de Welch.
# Passem el resultat obtingut al test d'abans a equal_var. Obtenim que les 
# mitjanes dels dos grups són iguals per SibSp.
print('\n--Test t de Student:', 
      '\tH0: les mitjanes són iguals',
      '\tH1: les mitjanes no són iguals', 
      sep = '\n')
alpha = 0.05
res = pd.DataFrame(columns = ['Variable','Estadistic','p-valor','H0'])
for var in ['Age','Fare','SibSp','Parch']:
    stat, p = ttest_ind(X0[var], X1[var],
                equal_var=res_homo.loc[(res_homo.Variable==var),'H0'].iloc[0])
    res = res.append({'Variable': var,
                      'Estadistic': stat,
                      'p-valor': round(p,6),
                      'H0': p>=alpha}, ignore_index = True)
print(res)

# ---- Mann-Whitney U test ----------------------------------------------------
# Una alternativa és utilitzar el Mann-Whitney U test, que ens dirà si les 
# medianes són iguales o no. Aquest mètode és més robust que l'anterior. 
# Obtenim que les medianes dels dos grups són iguals només per Age.
print('\n--Test U de Mann-Whitney:', 
      '\tH0: és igual de probable que un element aleatori de la mostra 1 '+
      '\n \t    sigui menor o major a un element aleatori de la mostra 2',
      '\tH1: les medianes de les dues mostres són diferents', 
      sep = '\n')
alpha = 0.05
res = pd.DataFrame(columns = ['Variable','Estadistic','p-valor','H0'])
for var in ['Age','Fare','SibSp','Parch']:
    stat, p = mannwhitneyu(X0[var], X1[var])
    res = res.append({'Variable': var,
                      'Estadistic': stat,
                      'p-valor': round(p,6),
                      'H0': p>=alpha}, ignore_index = True)
print(res)

# =============================================================================
# ~ Anàlisi - Correlacions
# =============================================================================
sep_("Anàlisi Correlacions")

# ---- Pearson ----------------------------------------------------------------
# El coeficient de de correlació Pearson ens indica si les variables mostren 
# una dependència lineal. -1 i +1 signifiquen completa linealitat mentre que 
# 0 significa que no hi ha cap correlació lineal. 
print('\n--Coeficient de correlació de Pearson:', 
      '\tH0: la correlació és nul·la',
      '\tH1: les variables no són independents linealment', 
      sep = '\n')
alpha = 0.05
res = pd.DataFrame(columns = ['Variable1', 'Variable2',
                              'Correlacio','Abs_corr','p-valor','H0'])
for var1 in X:
    for var2 in X:
        r, p = pearsonr(X[var1], X[var2])
        res = res.append({'Variable1': var1,
                          'Variable2': var2,
                          'Correlacio': r,
                          'Abs_corr': np.abs(r),
                          'p-valor': round(p,6),
                          'H0': p>=alpha}, ignore_index = True)
        
# Mostrem els valors amb una correlació mínima (0.5) i quants parells accepten
# H0 (són independents)
var = []
res2 = pd.DataFrame(columns=res.columns)
res3 = pd.DataFrame(columns=res.columns)
for _,row in res.iterrows(): 
    # Creem sets per no repetir parells
    s = {row.Variable1,row.Variable2}
    if s not in var:
        var.append(s)
        if (0.5<=row.Abs_corr) & (row.Variable1 != row.Variable2):
            res2 = res2.append(row, ignore_index=True)
        elif row.H0:
            res3 = res3.append(row, ignore_index=True)
            
# Veiem com és lògic, que diferents variables Title_X es relacionen amb el 
# sexe (Sex_male). També veiem que Pclass, Fare i Cabin_any estan una mica
# correlacionats entre ells per parelles. Finalment, Survived es troba 
# correlacionat amb el Sex_male i Title_Mr.
print('Valors amb correlació major o igual a 0.5:',
      res2.sort_values('Abs_corr', ascending = False), sep = '\n')
print('Total parells (no repetits) de variables significativament '+
      'independents: '+ str(res3.shape[0]) +'\nTop 5 per p-valor:')
print(res3.sort_values('p-valor', ascending = False).head(5))

# Heatmap de correlació
corr = res.pivot(columns='Variable1', index='Variable2', values='Correlacio')
create_heatmap(corr, vmin = -1, vmax = 1, cmap = 'coolwarm', 
               num = 'Pearson', figsize = (16,12))

# ---- Spearman ---------------------------------------------------------------
# El coeficient de de correlació Spearman ens indica si les variables mostren 
# una dependència monòtona. -1 i +1 signifiquen una funció perfectament 
# monòtona mentre que 0 significa que no. 
print('\n--Coeficient de correlació de Spearman:', 
      '\tH0: la correlació és nul·la',
      '\tH1: les variables mostren un comportament monoton', 
      sep = '\n')
alpha = 0.05
res = pd.DataFrame(columns = ['Variable1', 'Variable2',
                              'Correlacio','Abs_corr','p-valor','H0'])
for var1 in X:
    for var2 in X:
        r, p = spearmanr(X[var1], X[var2])
        res = res.append({'Variable1': var1,
                          'Variable2': var2,
                          'Correlacio': r,
                          'Abs_corr': np.abs(r),
                          'p-valor': round(p,6),
                          'H0': p>=alpha}, ignore_index = True)
        
# Mostrem els valors amb una correlació mínima (0.5) i quants parells accepten
# H0 (són independents)
var = []
res2 = pd.DataFrame(columns=res.columns)
res3 = pd.DataFrame(columns=res.columns)
for _,row in res.iterrows(): 
    # Creem sets per no repetir parells
    s = {row.Variable1,row.Variable2}
    if s not in var:
        var.append(s)
        if (0.5<=row.Abs_corr) & (row.Variable1 != row.Variable2):
            res2 = res2.append(row, ignore_index=True)
        elif row.H0:
            res3 = res3.append(row, ignore_index=True)
            
# Veiem com és lògic, que diferents variables Title_X es relacionen amb el 
# sexe (Sex_male). També veiem que Pclass, Fare i Cabin_any estan una mica
# correlacionats entre ells per parelles. Finalment, Survived es troba 
# correlacionat amb el Sex_male i Title_Mr.
print('Valors amb correlació major o igual a 0.5:',
      res2.sort_values('Abs_corr', ascending = False), sep = '\n')
print('Total parells (no repetits) de variables significativament '+
      'independents: '+ str(res3.shape[0]) +'\nTop 5 per p-valor:')
print(res3.sort_values('p-valor', ascending = False).head(5))

# Heatmap de correlació
corr = res.pivot(columns='Variable1', index='Variable2', values='Correlacio')
create_heatmap(corr, vmin = -1, vmax = 1, cmap = 'coolwarm', 
               num = 'Spearman', figsize = (16,12))

# =============================================================================
# ~ Anàlisi - Regressions
# =============================================================================
sep_("Anàlisi Regressions")

# Crearem una regressió logística per a prediure Survived. Utilitzarem les 
# variables explicatives que han mostrat força correlació amb Survived 
# (Sex_male o Title_Mr, però no les dues alhora). No utilitzarem les variables 
# que han mostrat independència significativa (Age i Embarked_Q). Provarem 
# d'afegir Fare, Pclass o Cabin_any però no alhora, ja que aquestes dues han 
# mostrat una mica de dependència.
print(res.loc[res.Variable1=='Survived']
      .sort_values('Abs_corr', ascending = False))

# La regressió logística ens permet prediure el resultat d'una variable 
# dicotòmica, com Survived. El resultat es dóna en un valor de 0 al 1 en el 
# qual podem determinar un llindar de separació. 
X = X.drop(['Survived'], axis= 1)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

print('\nRegressió logística:')
num = 'ROC_LR'
fig, axes = plt.subplots(2,4,figsize=(14,10),num = num)
for x, axs in zip([['Sex_male'],['Title_Mr']], axes):
    for y,ax in zip([[],['Fare'], ['Pclass'], ['Cabin_any']], axs):
        var = x + y
        clf = LogisticRegression(random_state = 42).fit(X_train[var],y_train)        
        y_prob = clf.predict_proba(X_test[var])[:,1]
    
# Per a avaluar el model, podem traçar una curva ROC (TPR vs FPR) segons els 
# diferents llindars. Com més propera a la cantonada superior esquerra, millor.
# També calculem l'àrea sota la corba (AUC): un valor de 1 és un model perfecte
# mentre que un valor de 0.5 equival a un model aleatori.
        tpr,fpr,scores = [[],[],[]]
        ths = np.arange(0,1.05,0.05)
        for th in ths[::-1]:
            y_pred = y_prob > th
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            tpr.append(tp / (tp+fn))
            fpr.append(fp / (fp+tn))
            scores.append((tn+tp) / (tp+tn+fn+fp))

# Integral per la regla dels trapezis per calucular auc        
        auc = 0
        for i in range(1,len(fpr)):
            auc += (tpr[i]+tpr[i-1]) / 2 * (fpr[i]-fpr[i-1]) 
            
        best = np.argmax(scores)
        print('\tVariables {:}, millor precisió: {:.4f} (llindar = {:.2f})'
              .format(var,scores[best],ths[best]))
         
        ax.plot(fpr,tpr,color='r')
        ax.plot([0,1],[0,1],color='grey', ls='--',lw=1)
        ax.set(title = 'Variables {:}\nAUC = {:.4f}'.format(var,auc))

plt.tight_layout()
plt.savefig(out_folder+'fig_{:s}.png'.format(num))
plt.show()

# =============================================================================
# ~ Anàlisi - Random Forest
# =============================================================================
sep_("Random Forest")

# Random Forests és un algorisme de tipus bagging que combina arbres de 
# decisió amb mostreigs aleatoris a través de votacions. Per a escollir els
# millors paràmetres, avaluem el conjunt de test amb una validació creuada.
clf = RandomForestClassifier(n_estimators=50, random_state=42)
par = {'max_depth' : range(2,12,2),
       'min_samples_split' : range(5, 55, 10)}
CV = GridSearchCV(clf, par, cv=5).fit(X_train, y_train)

res = pd.DataFrame(CV.cv_results_).pivot(columns='param_min_samples_split',
                                         index='param_max_depth',
                                         values='mean_test_score')    
create_heatmap(res, vmin = res.min().min(), vmax = res.max().max(), 
               cmap = 'cividis', num = 'CV_Random')

# Avaluem el conjunt de test amb el millor model
clf = CV.best_estimator_.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mat = confusion_matrix(y_test, y_pred)
mat = pd.DataFrame(mat).rename_axis(index='True',columns='Pred')
print('Millor classificador: max_depth={:d}, min_samples_split={:d}'
      .format(clf.max_depth, clf.min_samples_split))
print('Matriu de confusió:',mat,sep='\n')
print('Precisió: {:.4f}'.format(clf.score(X_test,y_test)))


# =============================================================================
# ~ Predicció test.csv
# =============================================================================

# ---- Regressió logística ----------------------------------------------------
var = ['Title_Mr', 'Pclass']
th = 0.5
clf = LogisticRegression(random_state = 42).fit(X_train[var],y_train)        
y_pred = (clf.predict_proba(test[var])[:,1] > th).astype(int)
survived = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
survived.to_csv( out_folder+'pred_LR.csv',  index=False)

# ---- Random Forest ----------------------------------------------------------
clf = CV.best_estimator_.fit(X_train, y_train)      
y_pred = clf.predict(test.drop('PassengerId',axis=1))
survived = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
survived.to_csv( out_folder+'pred_RF.csv',  index=False)