def remplaceinferieurpar0(valeur):
    """remplace ce qui commence par < par un 0.0
    
    Parameters : valeur (void)
    
    Returns:
    void or float : valeur or 0.0"""
    string=str(valeur)
    if string[0]=="<":
        return 0.0
    else:
        return valeur
    
#test fonction
#remplaceinferieurpar0("<3543543")
#remplaceinferieurpar0("3546387463")

def multi1000(valeur):
    """retourne une valeur multiplier par 1000
    
    Parameters:
        valeur:float
        
    Returns:
        valeur*1000:float
    """
    return valeur*1000

def changement_annee(str_annee):
    """retourne xxxx+1 au lieu de 'xxxx-yyyy'
    
    Parameters:
        str_annee:string
        
    Returns:
        str_annee:string"""
    return int(str_annee[0:4])+1
    
def Q6conv(valeur):
    """ retourne une valeur divisée par 1000
    Parameters:
        valeur:float
        
    Returns:
        valeur*1000:float"""
    return valeur/1000

if __name__ == "__main__" :
    print("##### test fonction ####")
    print("##### test fonction remplaceinferieurpar0 #######")
    print("valeur = '<3543543'")  
    print(remplaceinferieurpar0("<3543543"))
    print("valeur = '3546387463'")
    print(remplaceinferieurpar0("3546387463"))
    print("#### test fonction multi1000 ####")
    print("valeur = 5")
    print(multi1000(5))
    print("#### test changemement_annee #####")
    print("valeur = '2012-2014'")
    print(changement_annee("2012-2014"))
    print("#### test Q6conv ####")
    print("valeur = 5000")
    print(Q6conv(5000))
