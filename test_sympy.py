import random
from sympy import *
from time import time
from py4j.java_gateway import JavaGateway

varnames = ['GATA3', 'IFNbR', 'IFNg', 'IFNgR', 'IL10', 'IL10R', 'IL12R', 'IL18R', 'IL4', 'IL4R', 'IRAK', 'JAK1', 'NFAT', 'SOCS1', 'STAT1', 'STAT3', 'STAT4', 'STAT6', 'Tbet', 'IFNb', 'IL12', 'IL18', 'TCR']
n = len(varnames)

vars = []
for v in varnames:
    globals().__setitem__(v, Symbol(v))
    vars.append(Symbol(v))


f2 = lambda x: (x[0] | x[1]) and (not (x[3] | x[6]))

def eval_f(x, f, vars):
    """Évalue la fonction f au vecteur x."""

    return int(f.subs({vars[i]:x[i] for i in range(n)}) == True)

def eval_f_curry(f):
    return lambda x: int(f.subs({vars[i]:x[i] for i in range(n)}) == True)

N = 1000

t0 = time()
for _ in range(N):
    x = random.choices([0,1], k = n)
    y = eval_f(x,f,vars)

print(time()-t0)
t0 = time()

for _ in range(N):
    x = random.choices([0,1], k = n)
    y = eval_f_curry(f)(x)

print(time()-t0)
t0 = time()

for _ in range(100*N):
    x = random.choices([0,1], k = n)
    y = f2(x)

print(time()-t0)