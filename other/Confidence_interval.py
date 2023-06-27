import numpy as np
import pandas as pd
import math

fit1 = pd.read_csv('pre_evaluated/fit 60.csv').values
fit2 = pd.read_csv('pre_evaluated/fit 70.csv').values
fit3 = pd.read_csv('pre_evaluated/fit 80.csv').values
fit4 = pd.read_csv('pre_evaluated/fit 90.csv').values
c_val1=[]
con_me1,con_me11=[],[]
for i in range(0,fit1.shape[1]):
    c_val1.append(fit1[0][i])
    c_val1.append(fit2[0][i])
    c_val1.append(fit3[0][i])
    c_val1.append(fit4[0][i])
    SEM=np.std(np.array(c_val1))/math.sqrt(4)
    ts=0.95
    con_me1.append(np.mean(c_val1)+(ts*SEM))
    con_me11.append(np.mean(c_val1)-(ts*SEM))
A=np.array(con_me1)
B=np.array(con_me11)
m = np.empty([2,6])
m[0, :] = [mm for mm in A]
m[1, :] = [mmm for mmm in B]

df = pd.DataFrame(m.transpose(),
                      columns=['1','2'],
                      index=['CMBO','SSO','SSA','WHO','SMO','HYBRID'])
s = df.to_markdown()
print(s)


