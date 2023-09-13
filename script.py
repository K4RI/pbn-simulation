from PBN_simulation import *


def ex_toymodel():
    """Toy model présenté dans l'article d'AbouJaoudé2006."""

    # varnames = init_vars(4)
    # f0 = BoolFunc(x1 | x3, varnames)
    # f1 = BoolFunc(x0 & x3 | x2, varnames)
    # f2 = BoolFunc(~x0 & ~x3, varnames)
    # f3 = BoolFunc(x3, varnames)
    # toymodel = PBN(title = 'Toy model',
    #                n = 4,
    #                indep = False,
    #                f = [[f0, f1, f2, f3]],
    #                c = [1],
    #                sync = True,
    #                p = 0,
    #                q = 0,
    #                varnames = varnames)

    file_to_PBN('examples\\toymodel.pbn')
    print(toymodel)
    toymodel.simulation(10, verb = True)
    toymodel.stationary_law(T = 100, N = 25, R = 200)
    toymodel.STG()
    toymodel.copy_PBN(sync = False).STG()



def ex_shmulevich2001():
    """Toy model présenté dans l'article de Shmulevich2001."""

    # varnames = init_vars(3)
    # f0_0 = BoolFunc(x1 | x2, varnames)
    # f0_1 = BoolFunc((x1 | x2) & (~((~x0) & x1 & x2)), varnames)
    # f1_0 = BoolFunc((x0 | x1 | x2) & (x0 | (~x1) | (~x2)) & ((~x0) | (~x1) | x2), varnames)
    # f2_0 = BoolFunc((x0 & (x1 | x2)) | ((~x0) & x1 & x2), varnames)
    # f2_1 = BoolFunc(x0 & x1 & x2, varnames)
    # examplePBN = PBN(title = 'Shmulevich PBN',
    #                  n = 3,
    #                  indep = True,
    #                  f = [[f0_0, f0_1], [f1_0], [f2_0, f2_1]],
    #                  c = [[0.6, 0.4], [1], [0.5, 0.5]],
    #                  sync = True,
    #                  p = 0,
    #                  q = 1,
    #                  varnames = varnames)

    examplePBN = file_to_PBN('examples\\examplePBN.pbn')
    print(examplePBN)
    examplePBN.simulation(10, verb = False)
    examplePBN.STG_PBN()
    examplePBN.stationary_law(T = 100, N = 25, R = 200)



def ex_mammaliancellcycle():
    """Modèle biologique étudié dans Fauré2006."""

    # varnames = init_vars(10)
    # f0 = BoolFunc(x0, varnames)
    # f1 = BoolFunc(((~x4) & (~x9) & (~x0) & (~x3)) | (x5 & (~x9) & (~x0)), varnames)
    # f2 = BoolFunc(((~x1) & (~x4) & (~x9)) | (x5 & (~x1) & (~x9)), varnames)
    # f3 = BoolFunc((x2 & (~x1)), varnames)
    # f4 = BoolFunc((x2 & (~x1) & (~x6) & (~(x7 & x8))) | (x4 & (~x1) & (~x6) & (~(x7 & x8))), varnames)
    # f5 = BoolFunc(((~x0) & (~x3) & (~x4) & (~x9)) | (x5 & (~(x3 & x4)) & (~x9) &(~x0)), varnames)
    # f6 = BoolFunc(x9, varnames)
    # f7 = BoolFunc(((~x4) & (~x9)) | (x6) | (x5 & (~x9)), varnames)
    # f8 = BoolFunc((~x7) | (x7 & x8 & (x6 | x4 | x9)), varnames)
    # f9 = BoolFunc((~x6) & (~x7), varnames)
    # cellcycle = PBN(title = 'Mammalian cell cycle',
    #                 n = 10,
    #                 indep = False,
    #                 f = [[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]],
    #                 c = [1],
    #                 sync = True,
    #                 p = 0,
    #                 q = 0,
    #                 varnames = varnames)

    cellcycle = file_to_PBN('examples\\cellcycle.pbn')
    cellcycle.simulation(8, verb = True)
    cellcycle.stationary_law(T=200, N=500, R=500, show_all = False)

    prio_attrs = ["1000100000", "1000101111", "1000001010", "1000000010",
                  "1000000110", "1000000100", "1010000100", "1010100100",
                  "1001100100", "1011100000", "1010100000", "1000100100",
                  "1000001110", "1000100011", "1000101011", "1001100000",
                  "1011000100", "1011100100"]

    cellcycle.copy_PBN(sync = False)\
             .stationary_law(T=200, N=500, R=500, show_all = False,
                             prio_attrs = prio_attrs)



def test_claudine(q=1):
    """Calcule la prévalence des attracteurs Th0 Th1 Th2 dans le modèle
    Th_23, ainsi que dans ses extensions comportant les voisines des
    fonctions de référence dans le diagramme de Hasse.
    Cf. Cury2019, Mendoza2006."""

    for name in ['Table0_p08.bnet',
                 'Table1A_fAll_d1_p08.bnet',
                 'Table1B_fAll_d1_p08_siblings.bnet',
                 'Table1C_fGATA3_p08.bnet',
                 'Table1D_fTbet_p08.bnet',
                 'Table1E_fIL4_p08.bnet',
                 'Table1F_fIL4R_p08.bnet'
                 ]:

        if name == 'Table0_p08.bnet':
            filename = 'Experiments_Th_model\Table0_p08.bnet'
            zeroes, ones = [-1, -2, -3, -4], [] # <----------

            # Mendoza page 15 : "IFN-β, IL-12, IL-18 and TCR do not have inputs [...], treated as constants [...] having a value of 0"
        else:
            filename = "Experiments_Th_model\simul_prob_original\\" + name
            zeroes, ones = [i for i in range(23) if i != 2], [2] # <----------
            # P-O 6.2 : "Starting from an initial state, in which all the components are inactive but IFNg"
            # dans Mendoza fig 3, activer IFNg fait bouger de Th0 à Th1

        # zeroes, ones = [], []
        zeroes, ones = [-1, -2, -3, -4], []
        # zeroes, ones = [i for i in range(23) if i != 2], [2]

        def approach_attrs(x):
            if x[0] == '1':
                return '10001100110000010100000', 2
            if x[18] == '1':
                return '00110000000001000010000', 1
            return '00000000000000000000000', 0

        pbn_claudine = file_to_PBN(filename = filename)
        pbn_claudine.zeroes = zeroes
        pbn_claudine.ones = ones
        pbn_claudine.stationary_law()
        pbn_claudine.copy_PBN(sync = False).stationary_law()

        # pbn_claudine.stationary_law2(approach_attrs = approach_attrs,
        #                         attr_colors = [(0,255,0), (255,0,0), (0,0,255)],
        #                         attr_names = ['Th0', 'Th1', 'Th2'],
        #                         T = 20, R = 1000)



def test_syntheticBN(n, k):

    gs = [generateBN(n, k, sync = True, v = False, f = False, p = 0),
          generateBN(n, k, sync = True, v = False, f = True, p = 0)]

    for g in gs:
        print(g)
        g.regulation_graph()
        gasync = g.copy_PBN(sync = False)
        if n <= 6: g.STG()
        g.stationary_law(T = 20, N = 5, R = 50, show_all = False)
        if n <= 6: gasync.STG(layout = nx.spring_layout)
        gasync.stationary_law(T = 20, N = 5, R = 50, show_all = False)


def test_syntheticPBN(m, n, k):
    gs = [generate_Random_PBN([m,1,m,1], n, k, indep = True, sync = True,
                                                             p = 0, q = .1),
          generate_Random_PBN(m, n, k, indep = False, sync = True,
                                                      p = 0, q = .1)]

    for g in gs:
        print(g)
        g.regulation_graph()
        g.simulation(10, verb = True)
        g.STG_PBN()
        g.copy_PBN(sync = False).STG_PBN()


def test_filesPBN():

    model = file_to_PBN('output\Table1A_fAll_d1_p08_newsyntax.pbn')
    print(model)
    model.PBN_to_file('Table1A_redux')


    gs = [generateBN(5, 3, sync = True, v = False, f = False, p = 0), # reg
          generateBN(8, 4, sync = True, v = False, f = True, p = 0), # reg signé
          generate_Random_PBN([2,3,2,1,1], 5, 3, indep = True), # PBN indep
          generate_Random_PBN(2, 5, 3, indep = False)] # PBN non-indep

    for g in gs:
        print(g)
        g.regulation_graph()
        g.PBN_to_file()
        g2 = file_to_PBN(filename = 'output\\' + g.title + '.pbn')
        print(g2)
        g.regulation_graph()


def test_extended_PBN():

    bn = file_to_PBN('output\Table0_p08.pbn')
    print(bn)
    pbn_ext = generate_Extended_PBN(bn, i_modifs = [0, 8, 9, 18])
    pbn_ext.PBN_to_file()

    bns = [generateBN(4, 3, sync = True, v = True, f = True, p = 0),
           generateBN(7, 4, sync = True, v = False, f = True, p = 0),
           generateBN(4, 2, sync = True, v = False, f = False, p = 0)]

    for bn in bns:
        print(bn)
        bn.regulation_graph()
        bn.PBN_to_file()
        pbn_ext = generate_Extended_PBN(bn, i_modifs=None, p_ref=0.8, dist=2, q=1)
        print(pbn_ext)
        pbn_ext.PBN_to_file()


##
# Modèle simple à 4 noeuds : STG synchrone, STG asynchrone, loi synchrone
ex_toymodel()

# PBN à tirages indépendants : STG synchrone, loi synchrone
ex_shmulevich2001()

# Modèle biologique à 10 noeuds : loi synchrone, loi asynchrone avec affichage des attrs synchrones
ex_mammaliancellcycle()

# 6 simulations en synchrone et asynchrone) des fonctions voisines dans le diagramme de Hasse
test_claudine()

# BN synthétique à n noeuds et k voisins : STG, loi
test_syntheticBN(4,2)

# PBN synthétique à n noeuds et m contextes : STG
test_syntheticPBN(2, 4, 2)

# Divers exemples de conversion entre objets PBN et fichiers .pbn
test_filesPBN()

# BN à régulation signée par défaut, et ses extensions en PBN par fonctions voisines
test_extended_PBN()


##

# for i in range(len(pbn_ext.fcts)):
#     for fij in pbn_ext.fcts[i]:
#         print(i, to_dnf(fij))

g = generateGraph(4, 2, v = False)
for f in [False, True]:
    if f: print('\n----- signé')
    else: print('\n----- non-signé')
    bn = generateBNfromGraph(g, f = f, sync = True, p = 0)
    bn.regulation_graph()
    if not f:
        pbn_rand = generatePBNfromGraph(g, 3, indep = False, sync = True, p = 0, q = .1)
    pbn_ext = generate_Extended_PBN(bn, p_ref = 0.8, dist = 1, part = 'poly', q = 1)

    print('\nbn :'); bn.STG()
    if not f:
        print('\npbn random :'); pbn_rand.STG_PBN()
    print('\npbn extended :'); pbn_ext.STG_PBN()

    #stationary_law2(...)


#TODO : test_claudine(q) avec 0<q<1

#TODO : pbn = generate_Random_PBN(2, 3, 2, indep = False, q=1); pbn.STG(pbn.fcts[0]); pbn.STG(pbn.fcts[1]); pbn.STG_PBN()

##

colors_def = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (0,255,255), (255,0,255)]

def fct_attrs(att):
    if not all([len(a)==1 for a in att]):
        raise Exception('pas tous des points stables')
    pts = [next(iter(s)) for s in att]
    approach_attrs = lambda x: (''.join(map(str, x)), pts.index(''.join(map(str, x))))
    # attr_colors = [tuple([random.randint(0,255) for _ in range(3)]) for _ in pts]
    attr_colors = colors_def[:len(pts)]
    attr_names = pts
    return approach_attrs, attr_colors, attr_names

##

flag = True
essai, thres = 0, 4000
while flag:
    if essai > thres: raise Exception("nombre d'essais excédé")
    essai += 1
    if essai%50==0: print(essai)
    # bn = generateBN(5, 3, sync = True, v = False, f = True, p = 0)
    bn = generateBN(4, 3, sync = True, v = True, f = True, p = 0)
    a1 = bn.STG(pre=2)[1]
    if True: # on veut qu'il n'y ait que 3 points fixes
        pbn_ext = generate_Extended_PBN(bn, p_ref = 0.6, dist = 10, part = 'div', q = 1)
        a2 = pbn_ext.STG_PBN(pre=2)[1]
        flag = False
        # flag = (a1 != a2)
        # flag = (len(a2) != 3)

print('%i essais' %essai)
bn.regulation_graph()
print('------- BN')
print(bn.str_functs())
a1 = bn.STG()
print('------- PBN')
print(pbn_ext.str_functs())
a2 = pbn_ext.STG_PBN()


##
bn.PBN_to_file('bn')
pbn_ext.PBN_to_file('pbn')


##
i = 12

if i==0:
    bn = file_to_PBN('output\\bn.pbn', regulated = True)
    pbn_ext = file_to_PBN('output\\pbn.pbn')
else:
    bn = file_to_PBN('output\\bn%i.pbn'%i, regulated = True)
    pbn_ext = file_to_PBN('output\\pbn%i.pbn'%i)

bn.regulation_graph()
bn.STG()
pbn_ext.STG_PBN()
pbn_ext2.STG_PBN()
# bn.copy_PBN(sync = False).STG()
# pbn_ext.copy_PBN(sync = False).STG_PBN()
#
##
R = 1000

a1 = bn.STG(pre=2)[1]
approach_attrs, attr_colors, attr_names = fct_attrs(a1)
bn.copy_PBN(sync = True).stationary_law2(approach_attrs, attr_colors, attr_names, T=25, R=R)
pbn_ext = generate_Extended_PBN(bn, p_ref = 0.6, dist = 12, part = 'div', q = 1)
pbn_ext.copy_PBN(sync = True).stationary_law2(approach_attrs, attr_colors, attr_names, T=25, R=R)

pbn_ext2 = generate_Extended_PBN(bn, p_ref = 0.6, dist = 31, part = 'div', q = 1)
pbn_ext2.stationary_law2(approach_attrs, attr_colors, attr_names, T=200, R=R)

print('----------')




