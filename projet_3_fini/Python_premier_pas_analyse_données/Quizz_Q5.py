from piece_quizz import *

capitalAB=1000
for a in range(0,1000000):
    if lance_piece(50)==TRUE:
        capitalAB+=jeuA()
    else:
        capitalAB+=jeuB(capitalAB)
print (capitalAB)