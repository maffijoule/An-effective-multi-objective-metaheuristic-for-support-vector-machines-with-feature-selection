import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def minmax_norm(input):
    aux=np.array(input)
    for j in range(len(aux[0])):
        maxj=max(aux[:,j])
        minj=min(aux[:,j])
        for i in range(len(aux)):
            aux[i,j]=(2*aux[i,j]-(maxj-minj))/(maxj-minj)
    return(aux)

def kfold(x,y,index):
    len_x=len(x)
    size=len_x//5

    x_folds=[x[size*i:size*(i+1)] for i in range(4)]+[x[size*4:]]
    y_folds=[y[size*i:size*(i+1)] for i in range(4)]+[y[size*4:]]

    x_test = x_folds[index]
    y_test = y_folds[index]

    x_folds.pop(index)
    y_folds.pop(index)

    x_train= np.concatenate(x_folds)
    y_train= np.concatenate(y_folds)

    return x_train, x_test, y_train, y_test

def shuffle(x,y):
    index=[i for i in range(len(x))]
    np.random.shuffle(index)

    x_shuffle=[x[i] for i in index]
    y_shuffle=[y[i] for i in index]
    return(x_shuffle,y_shuffle)

def gisette(index):
    x = [[float(j) for j in i.split(' ') if j!=''] for i in open('Input/GISETTE/gisette_train.data','r').read().split('\n')]
    x_valid = [[float(j) for j in i.split(' ') if j!=''] for i in open('Input/GISETTE/gisette_valid.data','r').read().split('\n')]
    
    x.pop(-1)
    x_valid.pop(-1)

    y = [int(j) for i in open('Input/GISETTE/gisette_train.labels','r').read().split('\n') for j in i.split(' ') if j!='']
    y_valid = [int(j)  for i in open('Input/GISETTE/gisette_valid.labels','r').read().split('\n') for j in i.split(' ') if j!='']

    x, x_valid, y, y_valid=np.array(x), np.array(x_valid), np.array(y), np.array(y_valid)

    x=np.concatenate([x, x_valid])
    y=np.concatenate([y, y_valid])

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)

    x, x_valid, y, y_valid = kfold(x,y,index)

    return x, x_valid, y, y_valid

def wdbc(index):
    #file=[i.split(',') for i in open(r'C:\Users\mathi\Proyectos\TesisIND\Input\WDBC\wdbc.data','r').read().split('\n')[:-1]]
    file=[i.split(',') for i in open('Input/WDBC/wdbc.data','r').read().split('\n')[:-1]]
    
    #original el M es 1 y B -1
    for i in range(len(file)):
        if(file[i][1]=='M'):
            file[i].pop(0)
            file[i][0]=1
        elif(file[i][1]=='B'):
            file[i].pop(0)
            file[i][0]=-1

    y=[i[0] for i in file]
    x=[[float(j) for j in i[1:]] for i in file]

    x=minmax_norm(x)
    
    x=np.array(x)
    y=np.array(y)

    X_train, X_test, y_train, y_test= kfold(x,y,index)

    return(X_train, X_test, y_train, y_test)

def ionosphere(index):
    #file=[i.split(',') for i in open(r'C:\Users\mathi\Proyectos\TesisIND\Input\IONOSPHERE\ionosphere.data','r').read().split('\n')[:-1]]
    file=[i.split(',') for i in open('Input/IONOSPHERE/ionosphere.data','r').read().split('\n')[:-1]]
    for i in range(len(file)):
        if(file[i][-1]=='g'):
            file[i][-1]=1
        elif(file[i][-1]=='b'):
            file[i][-1]=-1
        file[i]=[float(j) for j in file[i]]
    y=[i[-1] for i in file]
    x=[i[:-1] for i in file]

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)
    
    #x=minmax_norm(x)   

    X_train, X_test, y_train, y_test= kfold(x,y,index)
    
    return(X_train, X_test, y_train, y_test)

def arcene(index):
    #x = [[float(j) for j in i.split(' ') if j!=''] for i in open(r'C:\Users\mathi\Proyectos\TesisIND\Input\ARCENE\arcene_train.data','r').read().split('\n')]
    #x_valid = [[float(j) for j in i.split(' ') if j!=''] for i in open(r'C:\Users\mathi\Proyectos\TesisIND\Input\ARCENE\arcene_valid.data','r').read().split('\n')]
    x = [[float(j) for j in i.split(' ') if j!=''] for i in open('Input/ARCENE/arcene_train.data','r').read().split('\n')]
    x_valid = [[float(j) for j in i.split(' ') if j!=''] for i in open('Input/ARCENE/arcene_valid.data','r').read().split('\n')]
    x.pop(-1)
    x_valid.pop(-1)
    #x=minmax_norm(x)
    #x_valid=minmax_norm(x_valid)

    #y = [int(j) for i in open(r'C:\Users\mathi\Proyectos\TesisIND\Input\ARCENE\arcene_train.labels','r').read().split('\n') for j in i.split(' ') if j!='']
    #y_valid = [int(j)  for i in open(r'C:\Users\mathi\Proyectos\TesisIND\Input\ARCENE\arcene_valid.labels','r').read().split('\n') for j in i.split(' ') if j!='']
    y = [int(j) for i in open('Input/ARCENE/arcene_train.labels','r').read().split('\n') for j in i.split(' ') if j!='']
    y_valid = [int(j)  for i in open('Input/ARCENE/arcene_valid.labels','r').read().split('\n') for j in i.split(' ') if j!='']

    x, x_valid, y, y_valid=np.array(x), np.array(x_valid), np.array(y), np.array(y_valid)

    x=np.concatenate([x, x_valid])
    y=np.concatenate([y, y_valid])

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)

    x, x_valid, y, y_valid = kfold(x,y,index)

    return x, x_valid, y, y_valid

def gc(index):
    file=[[float(j) for j in i.split(' ') if j!=''] for i in open('Input/GC/german.data-numeric','r').read().split('\n')][:-1]

    x = [i[:-1] for i in file]
    y = [int(i[-1]) if i[-1]==1 else -1 for i in file]

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)

    x=np.array(x)
    y=np.array(y)

    X_train, X_test, y_train, y_test= kfold(x,y,index)

    return(X_train, X_test, y_train, y_test)

def bioresponse(index):
    file= [[float(j) for j in i.split(',')] for i in open('Input/BIORESPONSE/bioresponse.arff','r').read().split('\n')[1781:-1]]

    x = [i[:-1] for i in file]
    y = [int(i[-1]) if i[-1]==1 else -1 for i in file]

    x=np.array(x)
    y=np.array(y)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)

    X_train, X_test, y_train, y_test= kfold(x,y,index)

    return(X_train, X_test, y_train, y_test)

def gina_agnostic(index):
    file= [[float(j) for j in i.split(',')] for i in open('Input/GAGNOSTIC/gina_agnostic.arff','r').read().split('\n')[997:-1]]

    x = [i[:-1] for i in file]
    y = [int(i[-1]) for i in file]

    x=np.array(x)
    y=np.array(y)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)

    X_train, X_test, y_train, y_test= kfold(x,y,index)

    return(X_train, X_test, y_train, y_test)

def duke(index):
    path = 'Input/DUKE/duke'
    file = [[float(j.split(':')[-1])  for j in i.split(' ') if j!=''] for i in open(path, 'r').read().split('\n')]

    x = [i[1:] for i in file]
    y = [int(i[0]) for i in file]

    x=np.array(x)
    y=np.array(y)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)

    X_train, X_test, y_train, y_test= kfold(x,y,index)

    return(X_train, X_test, y_train, y_test)

def madelon(index):
    x = [[float(j) for j in i.split(' ') if j!=''] for i in open('Input/MADELON/madelon_train.data','r').read().split('\n')]
    x_valid = [[float(j) for j in i.split(' ') if j!=''] for i in open('Input/MADELON/madelon_train.data','r').read().split('\n')]

    x.pop(-1)
    x_valid.pop(-1)

    y = [int(j) for i in open('Input/MADELON/madelon_train.labels','r').read().split('\n') for j in i.split(' ') if j!='']
    y_valid = [int(j)  for i in open('Input/MADELON/madelon_valid.labels','r').read().split('\n') for j in i.split(' ') if j!='']

    x, x_valid, y, y_valid=np.array(x), np.array(x_valid), np.array(y), np.array(y_valid)

    x=np.concatenate([x, x_valid])
    y=np.concatenate([y, y_valid])

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x,y = shuffle(x,y)
    x=np.array(x)
    y=np.array(y)

    x, x_valid, y, y_valid = kfold(x,y,index)

    return x, x_valid, y, y_valid

def mio():
    file = [[int(j) for j in i.split(' ') if j!=''] for i in open(r'C:\Users\mathi\Proyectos\TesisIND\Input\Mio\pruebas.data','r').read().split('\n')]
    x= [i[:2] for i in file]
    y= [i[-1] for i in file]

    x, x_valid, y, y_valid=np.array(x), np.array(x), np.array(y), np.array(y)
    return x, x_valid, y, y_valid
