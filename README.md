# Pràctica 2: Neteja i anàlisi de dades

## Descripció

Aquesta pràctica de neteja i anàlisi de dades ha estat realitzada pel Guillermo Camps Pons (individual) per a l'assignatura de _Tipologia i cicle de vida de les dades_ del Màster de Ciència de Dades de la Universitat Oberta de Catalunya (UOC). En aquesta pràctica, es tracten les dades del _dataset Titanic: Machine Learning from Disaster_, més concretament, els arxius _train.csv_ i _test.csv_ que es poden obtindre a [la competició de _Kaggle_](https://www.kaggle.com/c/titanic).

El programa _data_clean.py_ (escrit en Python 3.7) realitza un procés de neteja sobre les dades dels arxius _train.csv_ i _test.csv_ i genera dos arxius amb les dades netejades _train_clean.csv_ i _test_clean.csv_. Després, el programa realitza un anàlisi des de diferents perspectives de les dades netejades de _train_clean.csv_. Addicionalment, genera figures del processos de neteja i anàlisi i arxius csv amb les prediccions del l'arxiu _test.csv_ anomenats _pred_LR.csv_ i _pred_RF.csv_.

## Requisits d'execució

El codi present a _data_clean.py_ fa ús dels següents mòduls:

* _numpy_: s'utilitza per a operacions amb arrays.

* _pandas_: s'utilitza per a crear i gestionar DataFrames.

* _matplotlib_: es fa ús de _matplotlib.pyplot_ per a representar plots.

* _scipy_: es fa ús de _scipy.stats_ per a realitzar diferents tests estadístics.

* _sklearn_: s'utilitza per a la creació de mètodes de predicció i per la imputació de valors.

* _os_: s'utilitza per a crear el directori de output

Si no es tenen alguns d'aquests mòduls instal·lats conjuntament amb Python, s'haurà d'utilitzar l'instal·lador de paquets de Python (pip) de la següent manera:

```
pip install [nom_modul]
```

on el [nom_mòdul] és l'utilitzat a la llista anterior.

## Fitxers i directoris

* _data_clean.py_: codi que realitza tot el contingut de la pràctica.

* _data_: directori que conté els arxius d'input del programa _train.csv_ i _test.csv_.

* _output_: directori que conté tots els arxius csv i les figures que genera el programa _data_clean.py_.

* _README.md_: aquest arxiu.

* _M2.951_20192_Pràctica2.pdf_: pdf amb les respostes de la pràctica.
