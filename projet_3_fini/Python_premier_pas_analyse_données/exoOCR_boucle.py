"""exoOCR boucle"""
def recursivité_print(phrase,repetition):
     if repetition==0:
         return 1
     print (phrase)
     repetition-=1
     recursivité_print(phrase,repetition)

for a in range (0,5):
    print ("OpenClassRooms est vraiment top !")

#avec while
iterateur=0
while iterateur!=5:
    iterateur+=1
    print ("OpenClassRooms est vraiment top !")

recursivité_print("OpenClassRooms est vraiment top !",5)