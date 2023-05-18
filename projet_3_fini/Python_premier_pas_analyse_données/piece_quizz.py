from pickle import FALSE, TRUE
from random import randint

def lance_piece(probabilité_pile):
    """retourne true si pile, false si face"""
    chance=randint(0,100)
    if chance<probabilité_pile:
        return TRUE
    else:
        
        return FALSE

def jeuA():
    """simulation de pile ou face pipauter (p=0.49 de faire pile)"""
    if lance_piece(49)==TRUE:
        return 1 #on gagne 1 euro
    else:
        return -1 #on perd un euro


def jeuB(capital):
    """ 2 piece trafiqué prob pile piece 1 0.09 prob pile piece 2 0.74
    si le capital est un multiple de 3 on lance la piece 1 sinon on lace la 2
    on gagne 1 euro si pile sinon on perd 1 euro"""
    if capital%3!=0:
        if lance_piece(74)==TRUE:
            return 1
        else:
            return -1
    else:
        if lance_piece(9):
            return 1
        else:
            return -1

if __name__ == "__main__":
    #test pour le quizz OPC
    capitalDepart=1000
    capitalA=capitalDepart
    capitalB=capitalDepart

    for a in range(0,1000):
        capitalA+=jeuA()
        capitalB+=jeuB(capitalB)                
    print ("resultat A: ", capitalA )
    print("resultat B:", capitalB )