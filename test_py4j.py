## Test du lancement de functionhood depuis Python (1)

from py4j.java_gateway import JavaGateway
import psutil
import subprocess
import time

for _ in range(4):
    proc = subprocess.Popen(["java", "-jar", "functionhood/target/FunctionHood-0.1.jar"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    gateway = JavaGateway()

    try:
        hd = gateway.entry_point.initHasseDiagram(3)
        print(hd.getSize())

        a = gateway.entry_point.getFormulaParentsfromStr("{{1,2},{1,3},{2,3}}", True)
        print(a)
        psutil.Process(proc.pid).children()[0].terminate()
    except Exception as e:
        print('Error : ' + str(e))
        time.sleep(1)
        subprocess.run(["stop_jar.bat"])#, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        break


## Test du lancement de functionhood depuis Python (2)
from PBN_simulation import *
import psutil
import subprocess
import time

for i in range(5): # pour init dans ce fichier
    globals().__setitem__(f'x{i}', Symbol(f'x{i}'))
init_vars(5) # pour init dans PBN_simulation


for _ in range(3):
    proc = subprocess.Popen(["java", "-jar", "functionhood/target/FunctionHood-0.1.jar"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    try:
        fct = (x0 & x1) | (x3 & x0)
        print('\n', fct)
        print(voisines_direct([fct], 'c'))
        psutil.Process(proc.pid).children()[0].terminate()
    except Exception as e:
        print('Error : ' + str(e))
        time.sleep(1)
        subprocess.run(["stop_jar.bat"])#, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


## Test du parsing des expressions pour la recherche de fonctions voisines
from PBN_simulation import str_signed, voisines_direct, voisines
from sympy import *

varnames = ['GATA3', 'IFNbR', 'IFNg', 'IFNgR', 'IL10', 'IL10R', 'IL12R', 'IL18R', 'IL4', 'IL4R', 'IRAK', 'JAK1', 'NFAT', 'SOCS1', 'STAT1', 'STAT3', 'STAT4', 'STAT6', 'Tbet', 'IFNb', 'IL12', 'IL18', 'TCR']

for v in varnames:
    globals().__setitem__(v, Symbol(v))

f = eval('(GATA3 | IFNbR) & ! (IFNgR | IL12R)'.replace('!', '~'))
print(f)
print(to_dnf(f, simplify = True), '\n')
print(list(globals())[-26:], '\n')
vois = voisines(f, dist=2)

for vo in vois:
    for v in vo: print(v)
    print()


## Test du calcul des coefficients pour les fonctions voisines à distance 1,2,3...
from sympy import var, Eq, solve

ns = [3,2,3]
dist, p_ref = len(ns), 0.8
c_pbn = [p_ref]
part = 'div'

if any(ns):
    if part == 'poly':
        # On détermine le r tel que les voisins à distance k auront un poids r**k
        x = var('x')
        sols = solve(Eq(sum([ns[i]*x**(i+1) for i in range(dist)]), 1),x)
        r = max(list(filter(lambda x: 'I' not in str(x), sols)))
        c_pbn = [p_ref]
        for k in range(dist):
            c_pbn += [round((1-p_ref) * r**(k + 1), 6)] * ns[k]

    if part == 'div':
        # On détermine le r tel que les voisins à distance k auront un poids r/k
        r = 1/(sum([ns[k] / (k + 1) for k in range(dist)]))
        c_pbn = [p_ref]
        for k in range(dist):
            c_pbn += [round((1-p_ref) * r/(k + 1), 6)] * ns[k]

    if part == 'equal':
        r = 1/(sum(ns))
        c_pbn = [p_ref]
        c_pbn += [round((1-p_ref) * r, 6)] * sum(ns)

print(c_pbn)