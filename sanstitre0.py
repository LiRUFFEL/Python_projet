# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:49:59 2020

@author: lizheng
"""

####Excersices
##Exo 1
i=1
while i<=40 :
    if (i%2==0):
        print(i)
    i+=1
    
##Exo 2
a="3"
b="4"
print(a+b)
c=a+b##c est un string

##forcer faire une addition plutot
print(int(a)+int(b))

##Exo 3
f = 1
for i in range(1,11):
    f = f*i
    print(f)
    
##Exo 5
list1 = [1, 2, 3]
list2 = [1, "Hello", 3.4]
list3 = []
for i in list1 & list2:
    list3.append(i)
    print(list3)
    
list1 = [1, 2, 3]
list2 = [1, "Hello", 3.4]
list3 = []
for i in list1 :
    if i in list2:
        list3.append(i)
print(list3)

##Exo 6

list_a=[1,2,3,3,2,1]
list_b=list_a[::-1]
list_a==list_b

##Exo 7


##Exo 8

def maj(List_Char):
    j=0
    for i in List_Char:
        if i in string.ascii_uppercase:
            j+=1
        
    if j==len(List_Char):
        return True
    else:
        return False
    
##Exo 9

def hascap(x):
    string_list=x.split()
    i=0
    while i<len(string_list):
        if string_list[i][0] in string.ascii_uppercase:
            print(string_list[i])
        i+=1
            
S_1="I am A student"
hascap(S_1)

##test it
    
S_1="I am A student"  
S_2=S_1.split()
i=0
while i < len(S_2):
    if S_2[i][0] in string.ascii_uppercase:
        print(S_2[i])
    i+=1
    

print(S_1) 

##Exo 10
def lignes(s):
    mots = s.split() 
    lignes = [''] 
    for m in mots: 
        m += " " 
        if len(lignes[-1])+len(m)<24:
            lignes[-1] += (m) 
        else: lignes.append(m) 
    
s = "Onze ans deja que cela passe vite Vous "
s += "vous etiez servis simplement de vos armes la " 
s += "mort n’eblouit pas les yeux des partisans Vous " 
s += "aviez vos portraits sur les murs de nos villes"  
lignes(s)
print(lignes(s))   

s = "Onze ans deja que cela passe vite Vous "
mots = s.split() 
lignes = [''] 
for m in mots: 
    m += " " 
print(m)
    
    
    if len(lignes[-1])+len(m)<24:
        lignes[-1] += (m) 
    else: lignes.append(m)
print(lignes)
    
my_ligne="I am here!"
my_list=my_ligne.split()
print(my_list)
' '.join(my_list)


##Exo 11
def renvoie_numero(s):
    
    l = []
    for t in s.split():
        try:
            l.append(float(t))
        except ValueError:
            pass
        
    print(l)
    

s="I am here to test if -40 is in and 45.9 is in as well!"
renvoie_numero(s)


##Exo 12

def extraction(adr):
    file=open()
      

    