from lectura_alcaraz import gisette, wdbc,ionosphere, arcene, gc, bioresponse, gina_agnostic, duke, madelon
from NSGAII_OV import NSGA_II_1v
import pandas as pd
import sys
import csv
import os

os.environ['OMP_NUM_THREADS'] = '1'
ds = int(sys.argv[1])
t = int(sys.argv[2])
s = int(sys.argv[3])
split= int(sys.argv[4])
experiment_name=sys.argv[5]

ds_name=['Ionoshpere', 'WDBC','Gissette','Arcene','GermanCredit', 'BioResponse', 'GinaAgnostic', 'Duke', 'Madelon']
ds_func=[ionosphere, wdbc, gisette, arcene, gc, bioresponse, gina_agnostic, duke, madelon]
datasets={ds_name[ds]:ds_func[ds]}

experiments=[]
model = NSGA_II_1v(ds_func[ds], ds_name[ds], t,s,split, experiment_name)
output=model.main()
output['Dataset']=ds
output['Time']=t
output['Seed']=s
output['Fold']=split
experiments.append(output)


df=pd.DataFrame(experiments)
column_order = ['Dataset', 'Time'] + [col for col in df.columns if col not in ['Time', 'Dataset']]
df = df[column_order]
df.to_csv(f'Genetico/Outputs/{experiment_name}/datos_{ds_name[ds]}_{str(t)}_{str(s)}_{str(split)}.csv', index=False)