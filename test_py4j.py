##
from py4j.java_gateway import JavaGateway

gateway = JavaGateway()

hd = gateway.entry_point.getHasseDiagram()
hd.setSize(3)
print(hd.getSize())

gateway.entry_point.printHello()

a = gateway.entry_point.getFormulaParentsfromStr("{{1,2},{1,3},{2,3}}", True)
print(a)

##
from PBN_simulation import str_signed, voisines_direct, voisines
from sympy import *

varnames = ['GATA3', 'IFNbR', 'IFNg', 'IFNgR', 'IL10', 'IL10R', 'IL12R', 'IL18R', 'IL4', 'IL4R', 'IRAK', 'JAK1', 'NFAT', 'SOCS1', 'STAT1', 'STAT3', 'STAT4', 'STAT6', 'Tbet', 'IFNb', 'IL12', 'IL18', 'TCR']

for v in varnames:
    globals().__setitem__(v, Symbol(v))

f = eval('(GATA3 | IFNbR) & ! (IFNgR | IL12R)'.replace('!', '~'))
print(f)
print(to_dnf(f, simplify = True), '\n')
print(list(globals())[-26:], '\n')
vois = voisines(f, dist=3)

for vo in vois:
    for v in vo: print(v)
    print()