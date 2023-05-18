from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt

def detection_clé(df):
    """indique si une colonne peut être utilisé en clé
    Args:
        df : dataframe as DataFrame
    
    Returns :
        None"""

    cle_non_detecte = True
    for x in df:
        if len(df[x].unique()) == df.shape[0]: #si le nombre de ligne est egal au nombre d'élement unique alors c 'une clé
            print("la colonne",x,"est une clé")
            cle_non_detecte = False
    if cle_non_detecte:
        print("pas de clé") 

def analyseDataFrame(df):
    """affiche un resumé des Dataframes info describe head et verification doublons

    Args:
        df : dataframe à analyser as dataFrame

    Returns:
        None"""

    print ("################################# INFO  #################################")
    df.info()
    print ("################################# DESCRIBE  #################################")
    display(df.describe(include='all',datetime_is_numeric=True))
    print ("################################# HEAD  #################################")
    display(df.head())
    print ("################################# DOUBLONS  #################################")
    display(df[df.duplicated()])
    print("################################## recherche de clé pour merge ###################")
    detection_clé(df)
    
def categoriser(data,nom_col_recherche,mot_cle,nom_categorie,nom_col_categorie='categorie',regex=False):
    """recherche un mot dans une string et enregistre un nom de categorie dans la colonne categorie (par defaut)
    Args:
        data : dataframe a categoriser as DataFrame
        nom_col_recherche : colonne où rechercher as string
        nom_categorie : nom de la catégorie as string
        nom_col_categorie : nom de la colonne cat"egorie a créer as string
    Returns :
        None"""
    data.loc[data[nom_col_recherche].str.contains(mot_cle,regex=regex) , nom_col_categorie] = nom_categorie

def courbe_Lorentz(plt_ax,df,colonne,titre="",titreX="",titreY=""):
    """configure une courbe de Lorentz sur l'axe matplotlib
    Args:
        plt_ax : Axe Matplotlib as matplotlib.axes_subplots.AxesSubplot
        df : dataframe a exploiter as DataFrame
        colonne: nom de la colonne pour faire le graphe as string
        titre : titre du graphe as string
        titreX : titre de l'axe X as string
        titreY: titre de l'axe Y as string
    Returns :
        None"""

    #classer les valeur en fonction du chiffre d'affaire
    data_lorentz = df.sort_values(by=colonne,ascending=False).copy(deep=True)

    #enlevé les Nan
    data_lorentz = data_lorentz[~data_lorentz[colonne].isna()]

    #création de colonnes pour le graphe de lorentz

    #CA cumulé
    data_lorentz['pourcentage Y cumulé'] = (data_lorentz[colonne]/data_lorentz[colonne].sum()*100).cumsum()

    #pourcentage de bouteille je me sers d'un nouvel index
    data_lorentz = data_lorentz.reset_index()
    data_lorentz = data_lorentz.drop(columns='index')

    data_lorentz['pourcentage index']=data_lorentz.index/(data_lorentz.shape[0]-1)*100

    
    # creation d'une courbe de Lorentz
    plt_ax.plot(data_lorentz['pourcentage index'],data_lorentz['pourcentage Y cumulé'])

    #courbe de lorentz avec repartition ideale
    plt_ax.plot([100, 0],[100,0])

    #titre,et titre des axes
    plt_ax.set_title(titre)
    plt_ax.set_xlabel(titreX)
    plt_ax.set_ylabel(titreY)    





######################################################## tests des fontions  #########################################################################

if __name__ == '__main__':
    df = pd.DataFrame({'A': [1.1, 2.7, 5.3],
                       'B': [2, 10, 9],
                       'C': [1.0, 5.4, 1.0],
                       'D': [4, 15, 15]},
                      index = ['a1', 'a2', 'a3'])
    lst = [["aliment sandwich",'aliment'],
           ["aliment Jambon","aliment"],
           ['outil clé a molette','aliment'],
           ['outil tournevis','aliment']]
    df2 = pd.DataFrame(lst, columns = ['Nom','type'])
    
    print("######### exemple d'analyseDataFrame ##########")
    analyseDataFrame(df)

    print("######### exemple detection clé dataframe ########")
    detection_clé(df)

    print("######### exemple categorize ########")
    print("##### dataframe d'exemple #######")
    print(df2.head())

    print("categoriser(df2,'Nom','outil','outilage','type')")
    categoriser(df2,'Nom','outil','outilage','type')
    print(df2.head())    