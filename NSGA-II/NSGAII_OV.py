import numpy as np
import seaborn as sns
from colour import Color
import bisect
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, f1_score, cohen_kappa_score, log_loss
from sklearn.model_selection import train_test_split
import os
import imageio


os.environ['OMP_NUM_THREADS'] = '1'

"""
--lb_f 0.699 --ub_f 0.828 
--lb_w -6.744 --ub_w 6.156 
--lb_b 4.576 --ub_b 6.514 
--pm_1 0.486 --pm_2 0.439
"""


class NSGA_II_1v():
    def __init__(self, dataset, name, time, seed, split, experiment,losses):
        np.random.seed(seed)
        self.losses=losses
        self.seed=seed
        #Variables a utilizar
        self.x, self.X_test, self.y, self.y_test= dataset(split)

        cont=0
        for i in self.y:
            if(i==1):
                cont+=1
        alpha=cont/len(self.y)
        self.alpha_coef=[alpha if i==1 else 1-alpha for i in self.y]

        self.split=split
        self.cont=0
        self.population={}
        self.obj={}
        self.pareto_ev={}
        self.times=[]
        #Parametros requeridos en otras funciones
        self.n_features=len(self.x[0])
        self.time=time
        self.name=name
        self.experiment_name=experiment
        if not os.path.exists(f'Genetico/Outputs/{self.experiment_name}'):
            os.makedirs(f'Genetico/Outputs/{self.experiment_name}')

        self.loss_val=[]

        self.adj_time={}
    
    def loss(self):
        y=self.y_test
        loss_mean=0
        cont=0
        mini=100000
        for i in self.population:
            if(self.pareto_f[i]==1):
                cont+=1
                ind=self.population[i]
                x=self.X_test[:,np.array([int(i) for i in ind[2]])]
                #y_pred = (probabilidades >=0).astype(int)
                y_pred = (np.dot(x, ind[0]) + ind[1])
                probabilidades = 1 / (1 + np.exp(-y_pred))
                ind_loss=log_loss(y,probabilidades)
                loss_mean+=ind_loss
                if(ind_loss<mini):
                    mini=ind_loss
        self.loss_val.append(loss_mean/cont)
        #self.loss_val.append(mini)
    
    def plot_loss(self):
        plt.figure() 
        x=[i for i in range(len(self.loss_val))]
        text=''
        for i in self.loss_val:
            text+=str(i)+' '
        plt.plot(x, self.loss_val, marker='o', linestyle='-')

        # Añadir etiquetas y título
        plt.xlabel('Iteracion')
        plt.ylabel('Loss_F')

        # Mostrar la gráfica
        plt.savefig(f'Genetico/Outputs/{self.experiment_name}/{self.name}_{str(self.time)}_{self.seed}.png')
        with open(f'Genetico/Outputs/{self.experiment_name}/{self.name}_{str(self.time)}_{self.seed}.txt', 'w') as file:
            file.write(text)

        text=''
        for i in self.times:
            text+=str(i)+' '
        with open(f'Genetico/Outputs/{self.experiment_name}/times_{self.name}_{str(self.time)}_{self.seed}.txt', 'w') as file:
            file.write(text)

    def pareto_historical(self):
        #Esta función guarda la iteracion de los individuos que estaban en la frontera de pareto
        for i in self.pareto_f:
            if(self.pareto_f[i]==1):
                self.pareto_ev[self.obj[i]]=self.iter

    def plot_gif(self):
        ph = self.pareto_ev
        # Crear el directorio temporal para los frames
        frames_dir = f'Genetico/Outputs/{self.experiment_name}/frames_{self.name}_{self.time}_{self.seed}_{self.split}'
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

        red = Color("blue")
        colors = list(red.range_to(Color("red"), max(ph.values()) + 1))

        sort_pareto = [[[], []] for i in range(max(ph.values()) + 1)]


        max_o1=0
        max_o2=0
        for i in ph:
            sort_pareto[ph[i]][0].append(i[0])
            sort_pareto[ph[i]][1].append(i[1])
            if(i[0]>max_o1):
                max_o1=i[0]
            if(i[1]>max_o2):
                max_o2=i[1]
        step=1
        if(len(sort_pareto)>300):
            step=len(sort_pareto)//200
        # Generar y guardar cada frame
        filenames = []

        for i in range(len(sort_pareto)):
            if(self.iter%step==0):
                f, ax = plt.subplots(1, 1, figsize=(12, 12))
                sns.scatterplot(x=sort_pareto[i][0], y=sort_pareto[i][1], ax=ax, marker=".", linewidth=1, color=colors[i].hex_l, s=300)
                ax.set_xlim(0, max_o1)  # Establecer el límite del eje X
                ax.set_ylim(0, max_o2) 
                ax.set_title(f'{self.name}_{str(self.time)}_{self.seed}')
                filename = f'{frames_dir}/frame_{i}.png'
                filenames.append(filename)
                plt.savefig(filename)
                plt.close(f)


        # Crear el GIF
        nombre= f'Genetico/Outputs/{self.experiment_name}/{self.name}_{str(self.time)}_{self.seed}_G.gif'
        with imageio.get_writer(nombre, mode='I', duration=0.1) as writer:
            for filename in filenames:
                try:
                    image = imageio.imread(filename)
                    writer.append_data(image)
                except:
                    pass

        # Limpiar los archivos temporales
        for filename in filenames:
            os.remove(filename)
        os.rmdir(frames_dir)

    def individual(self):
        f=np.random.randint(4*self.n_features//5, self.n_features)

        sv=[np.random.uniform(-3,3) for i in range(f)]
        b=np.random.uniform(-5,5)
        f = np.sort(np.random.choice(self.n_features, f, replace=False)).tolist()


        ind=[sv,b,f]
        f1,f2=self.evaluation(ind)
        self.population[self.cont]=ind
        self.obj[self.cont]=(f1,f2)
        self.cont+=1

    def evaluation(self, ind):
        sv=ind[0]
        b=ind[1]
        f1=np.sum([i**2 for i in sv])*0.5

        select=np.array([int(i) for i in ind[2]])
        try:
            x=self.x[:,select]
        except:
            print(select)
            x=self.x[:,select]

        rows=np.dot(x,sv)
        rows=rows+np.full(shape=rows.shape, fill_value=b)
        dot=self.y*self.alpha_coef*rows
        e=1-dot
        aux=np.where(e>=0,e,0)
        #e=np.sum(aux)/self.n_features
        e=np.sum(aux)
        return(f1,e)

    def low_frontier(self):
        self.index=[]
        front=1
        while(len(self.index)<self.n_ind//2): #Esto estaba partido en 10 al 22-06
            aux=[i for i in self.population if self.pareto_f[i]==front]
            self.index+=aux
            front+=1

    def child(self, ind1, ind2):
        splits=3 #hiperparametro
        size=len(ind1[2])//splits
        if(size<=1):
            size=2
        
        b=(ind1[1]+ind2[1])/2

        w=ind1[0][:size]
        f=ind1[2][:size]

        inicio=f[-1]
        container=[ind1,ind2]
        for i in range(1,splits): 
            if(i%2==0):
                parent=container[0]
            else:
                parent=container[1]
                
            i_pos = next((i for i, x in enumerate(parent[2]) if x > inicio), None)
            
            if(i_pos==None):
                continue

            f_pos = i_pos+size
            
            w+=parent[0][i_pos:f_pos]
            nf=parent[2][i_pos:f_pos]
            f+=nf
            inicio = f[-1]
        if(len(w)<4):
            print('what?')
        return([w,b,f])

    def crossover(self):
        p1,p2=np.random.choice(self.index,2)

        if(self.pareto_f[p2]<self.pareto_f[p1]):
            aux=p1
            p1=p2
            p2=aux
        elif(self.pareto_f[p2]==self.pareto_f[p1]):
            if(self.S[p1]<=self.S[p2]):
                aux=p1
                p1=p2
                p2=aux

        ind=self.child(self.population[p1],self.population[p2])

        ind=self.mutation(ind)
        ind=self.mutation_nf(ind)
        f1,f2=self.evaluation(ind)
        self.population[self.cont]=ind
        self.obj[self.cont]=(f1,f2)
        self.cont+=1

    def mutation(self,ind):
        if(len(ind[0])==0):
            print('perdido')
        if(np.random.random()<0.6*0.999**self.iter):
            if(np.random.random()<(1/3)):
                pos=np.random.randint(0,len(ind[0])-1)
                amount=np.random.uniform(-0.05,0.05)
                ind[0][pos]=ind[0][pos]+ind[0][pos]*amount
            if(np.random.random()<(1/3)):
                b=np.random.uniform(-0.1,0.1)
                ind[1]=ind[1]+b*ind[1]
            if(len(ind[2])!=self.n_features and np.random.random()<(1/3)):
                pos=np.random.randint(0,len(ind[2])-1)
                new_f=np.random.randint(0,self.n_features)
                while(new_f in ind[2]):
                    new_f=np.random.randint(0,self.n_features)
                ind[2].pop(pos)
                bisect.insort(ind[2],new_f)
        return(ind)

    def binary_search(self, arr, target):
        left = 0
        right = len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return False
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return True

    def mutation_nf(self, ind):
        if(np.random.random()<0.3*0.999**self.iter):
            f=len(ind[2])
            if(np.random.random()>=0.5):
                if(len(ind[2])<self.n_features-2):
                    n=np.random.randint(1,max(2,f//5))
                    while(f+n>self.n_features):
                        n=np.random.randint(1,max(2,f//5))
                    av_features=[i for i in range(self.n_features) if self.binary_search(ind[2],i)] #available
                    for i in av_features:
                        pos=bisect.bisect_left(ind[2], i)
                        ind[2].insert(pos,i)
                        ind[0].insert(pos, np.random.uniform(-3,3))
            elif(f>7):
                n=np.random.randint(1,max(2,f//5))
                p=n/f
                delete = [0 if np.random.random() > p else 1 for i in range(f)]
                for i in range(len(delete)-1,-1,-1):
                    if(delete[i]==1):
                        ind[0].pop(i)
                        ind[2].pop(i)
        return(ind)

    def pareto(self):
        P=self.obj
        self.pareto_f={}
        n={}
        S={}
        F=[[]]
        for p in P:
            n_p=0
            Sp=[]
            for q in P:
                #Aqui establecemos el criterio de seleccion para min min. ahora esta min min
                if(P[p][0]<P[q][0] and P[p][1]<P[q][1]):
                    Sp.append(q)
                elif(P[p][0]>P[q][0] and P[p][1]>P[q][1]):
                    n_p+=1
            if(n_p==0):
                F[0].append(p)
            else:
                n[p]=n_p
            S[p]=tuple(Sp)
        i=0
        while(F[i]):
            Q=[]
            for p in F[i]:
                for q in S[p]:
                    n[q]-=1
                    if(n[q]==0):
                        Q.append(q)
            i+=1
            F.append(Q)
        for i in range(len(F)):
            for j in F[i]:
                self.pareto_f[j]=i+1
        self.S=S

    def crowding_distance(self, front):
        num_solutions = len(front)
        num_objectives = len(front[0])
        
        distances = np.zeros(num_solutions)
        
        for m in range(num_objectives):
            # Sort solutions based on this objective
            sorted_indices = np.argsort([sol[m] for sol in front])
            sorted_front = [front[i] for i in sorted_indices]
            
            # Set the distance to infinity for boundary solutions
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float('inf')
            
            # Get the min and max values for this objective
            min_obj = sorted_front[0][m]
            max_obj = sorted_front[-1][m]
            
            # Avoid division by zero
            if max_obj == min_obj:
                continue
            
            # Calculate crowding distance for each solution in the middle
            for i in range(1, num_solutions-1 ):
                distances[sorted_indices[i]] += (sorted_front[i + 1][m] - sorted_front[i-1][m]) / (max_obj - min_obj)

        return distances

    def selection_(self):
        new_pop={}
        new_obj={}
        front=1
        while(len(new_pop)<self.n_ind):
            for ind in self.population:
                if(self.pareto_f[ind]==front and len(new_pop)<self.n_ind):
                    new_pop[ind]=self.population[ind]
                    new_obj[ind]=self.obj[ind]
            front+=1
        self.population={}
        self.obj={}
        self.population=new_pop
        self.obj=new_obj

    def selection(self):
        new_pop={}
        new_obj={}
        front=1
        while(len(new_pop)<self.n_ind):
            aux=[]
            for ind in self.population:
                if(self.pareto_f[ind]==front):
                    aux.append(ind)
            if(len(new_pop)+len(aux)<=self.n_ind):
                for ind in aux:
                    new_pop[ind]=self.population[ind]
                    new_obj[ind]=self.obj[ind]
            else:
                front_objs=[self.obj[i] for i in aux]
                distances=self.crowding_distance(front_objs)
                distances=[(aux[i],distances[i]) for i in range(len(distances))]
                distances=sorted(distances, key=lambda x: x[1], reverse=True)
                for i in range(self.n_ind-len(new_pop)):
                    ind=distances[i][0]
                    new_pop[ind]=self.population[ind]
                    new_obj[ind]=self.obj[ind]

            front+=1
        self.population={}
        self.obj={}
        self.population=new_pop
        self.obj=new_obj

    def ploteo(self, itera):
        plt.figure(itera)
        puntos=[]
        par=[]
        for i in self.obj:
            if (self.pareto_f[i]==1):
                par.append(self.obj[i])
            elif(self.pareto_f[i]<=10):
                puntos.append(self.obj[i])
        # Creating a numpy array
        X = np.array([i[0] for i in puntos])
        Y = np.array([i[1] for i in puntos])
        x = np.array([i[0] for i in par])
        y = np.array([i[1] for i in par])

        X= [2/(2*i)**(0.5) for i in X]
        x= [2/(2*i)**(0.5) for i in x]
        # Plotting point using scatter method
        plt.scatter(X,Y)
        plt.scatter(x,y)
        itera=str(itera)
        while(len(itera)!=3):
            itera='0'+itera
        name=r'C:\Users\mathi\Proyectos\Tesis IND\Genetico\Outputs\ploteo_'+itera+'.png'
        plt.title(itera)
        #plt.show()
        plt.savefig(name)

    def export(self):
        ruta=r'C:\Users\mathi\Proyectos\svm_genetico\Output_V1_3'
        popfile=open(ruta+'\pop_free_output.txt','w')
        popfile.write(str(self.population))
        popfile.close()
        objfile=open(ruta+'\obj_free_output.txt','w')
        objfile.write(str(self.obj))
        objfile.close()
        parfile=open(ruta+'\par_free_output.txt','w')
        parfile.write(str(self.pareto_f))
        parfile.close()

    def auc(self, ind ):
        x=self.X_test[:,ind[2]]
        y=self.y_test
        scores = np.dot(x, ind[0]) + ind[1]

        # Calcular las probabilidades utilizando la función de regresión logística (sigmoid)
        probabilidades = 1 / (1 + np.exp(-scores))
        # Calcular el AUC
        auc = roc_auc_score(y, probabilidades)

        #y_pred = (probabilidades >=0).astype(int)
        y_pred=scores/(abs(scores))
        accurracy = accuracy_score(y, y_pred)
        if(accurracy<0.5):
            accurracy=1-accurracy
        fsc=f1_score(y, y_pred)
        ck=cohen_kappa_score(y, y_pred)
        return(auc, accurracy, fsc, ck)
    
    def main(self):
        print('Inicio')
        self.n_ind=50
        iterations=20
        for i in range(self.n_ind):
            self.individual()
        self.pareto()
        self.low_frontier()
        self.iter=0
        past=time.time()

        output={}

        while(True):
            #Crossover and mutatiom
            for j in range(self.n_ind):
                self.crossover()

            #Pareto and ploting
            self.pareto()
            self.pareto_historical()

            #Selection
            self.selection()
            self.low_frontier()
            
            if(self.losses):
                self.loss()
            self.times.append(time.time()-past)
            if(time.time()-past>self.time):
                break

            self.iter+=1
        print('Saliendo: ',i)
        self.export()
        aucs=[]
        acurracys=[]
        fscs=[]
        ck=[]
        prints=[]
        self.plot_loss()
        self.plot_gif()
        for i in self.population:
            if(self.pareto_f[i]==1):
                ind=self.population[i]
                out=self.auc(ind)
                aucs.append(out[0])
                acurracys.append(out[1])
                fscs.append(out[2])
                ck.append(out[3])
                prints.append([len(ind[2]), out])
                if(len(ind[2])>len(self.X_test[0])):
                    print('alertaaaaaaaaaaaaa')
            
        output['AUC']=np.max(aucs)
        output['FSC']=np.max(fscs)
        output['CohenKappa']=np.max(ck)
        
        output['AUC_mean']=np.mean(aucs)
        output['AUC_std']=np.std(aucs)
        output['Features_auc']=prints[np.argmax(aucs)][0]
        output['Accurracy_max']=np.max(acurracys)
        output['Accurracy_mean']=np.mean(acurracys)
        output['Accurracy_std']=np.std(acurracys)
        output['Features_acc']=prints[np.argmax(acurracys)][0]
        output['CohenKappa_mean']=np.mean(ck)
        output['CohenKappa_std']=np.std(ck)
        output['Features_ck']=prints[np.argmax(ck)][0]
        output['Iteraciones']=self.iter

        print('Fin')
        
        return(output)