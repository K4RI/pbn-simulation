from PBN_simulation import *


def ex_toymodel():
    """Toy model présenté dans l'article d'AbouJaoudé2006."""
    varnames = init_vars(4)
    print(varnames)

    f0 = eval('x1 | x3')
    f1 = eval('x0 & x3 | x2')
    f2 = eval('~x0 & ~x3')
    f3 = eval('x3')

    toymodel = PBN(title = 'Toy model',
                   n = 4,
                   indep = False,
                   f = [[f0, f1, f2, f3]],
                   c = [1],
                   sync = True,
                   p = 0,
                   q = 0,
                   varnames = varnames)
    print(toymodel)
    toymodel.simulation(10, verb = True)
    toymodel.stationary_law(T = 20, N = 5, R = 50)
    toymodel.STG()
    toymodel.copy_PBN(sync = False).STG()



def ex_shmulevich2001():
    """Toy model présenté dans l'article de Shmulevich2001."""
    varnames = init_vars(3)

    f0_0 = eval('x1 | x2')
    f0_1 = eval('(x1 | x2) & (~((~x0) & x1 & x2))')
    f1_0 = eval('(x0 | x1 | x2) & (x0 | (~x1) | (~x2)) & ((~x0) | (~x1) | x2)')
    f2_0 = eval('(x0 & (x1 | x2)) | ((~x0) & x1 & x2)')
    f2_1 = eval('x0 & x1 & x2')

    examplePBN = PBN(title = 'Shmulevich PBN',
                     n = 3,
                     indep = True,
                     f = [[f0_0, f0_1], [f1_0], [f2_0, f2_1]],
                     c = [[0.6, 0.4], [1], [0.5, 0.5]],
                     sync = True,
                     p = 0,
                     q = 1,
                     varnames = varnames)
    print(examplePBN)
    examplePBN.simulation(10, verb = False)
    examplePBN.STG_PBN()
    examplePBN.stationary_law(T = 20, N = 5, R = 50)



def ex_mammaliancellcycle():
    """Modèle biologique étudié dans Fauré2006."""
    varnames = init_vars(10)

    f0 = x0

    f1 = eval('((~x4) & (~x9) & (~x0) & (~x3)) | (x5 & (~x9) & (~x0))')

    f2 = eval('((~x1) & (~x4) & (~x9)) | (x5 & (~x1) & (~x9))')

    f3 = eval('(x2 & (~x1))')

    f4 = eval('(x2 & (~x1) & (~x6) & (~(x7 & x8))) '
                 '| (x4 & (~x1) & (~x6) & (~(x7 & x8)))')

    f5 = eval('((~x0) & (~x3) & (~x4) & (~x9)) | (x5 & (~(x3 & x4)) & (~x9) &(~x0))')

    f6 = eval('x9')

    f7 = eval('((~x4) & (~x9)) | (x6) | (x5 & (~x9))')

    f8 = eval('(~x7) | (x7 & x8 & (x6 | x4 | x9))')

    f9 = eval('(~x6) & (~x7)')

    cellcycle = PBN(title = 'Mammalian cell cycle',
                    n = 10,
                    indep = False,
                    f = [[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]],
                    c = [1],
                    sync = True,
                    p = 0,
                    q = 0,
                    varnames = varnames)
    cellcycle.simulation(8, verb = True)
    cellcycle.stationary_law(T = 50, N = 50, R = 100, show_all = False)

    prio_attrs = ["1000100000", "1000101111", "1000001010", "1000000010",
                  "1000000110", "1000000100", "1010000100", "1010100100",
                  "1001100100", "1011100000", "1010100000", "1000100100",
                  "1000001110", "1000100011", "1000101011", "1001100000",
                  "1011000100", "1011100100"]

    cellcycle.copy_PBN(sync = False)\
             .stationary_law(T = 50, N = 50, R = 100, show_all = False,
                             prio_attrs = prio_attrs)



def test_claudine():
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
                 'Table1F_fIL4R_p08.bnet']:

        if name == 'Table0_p08.bnet':
            filename = 'Experiments_Th_model\Table0_p08.bnet'
            zeroes, ones = [-1, -2, -3, -4], []
        else:
            filename = "Experiments_Th_model\simul_prob_original\\" + name
            zeroes, ones = [i for i in range(23) if i != 2], [2]

        def approach_attrs(x):
            if x[0] == '1':
                return '10001100110000010100000', 2
            if x[18] == '1':
                return '00110000000001000010000', 1
            return '00000000000000000000000', 0

        pbn_claudine = file_to_PBN(filename = filename,
                                   zeroes = zeroes, ones = ones, sync = True)
        pbn_claudine.stationary_law2(approach_attrs = approach_attrs,
                                attr_colors = [(0,255,0), (255,0,0), (0,0,255)],
                                attr_names = ['Th0', 'Th1', 'Th2'],
                                T = 20, R = 100)



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


def test_syntheticPBN(m, n, k, indep):
    gs = [generate_Random_PBN([m,1,m,1], n, k, indep = True, sync = True,
                                                             p = 0, q = .1),
          generate_Random_PBN(m, n, k, indep = False, sync = True,
                                                      p = 0, q = .1)]

    for g in gs:
        print(g)
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

    bns = [generateBN(5, 2, sync = True, v = False, f = True, p = 0),
          generateBN(9, 4, sync = True, v = False, f = True, p = 0)]

    for bn in bns:
        print(bn)
        bn.regulation_graph()
        bn.PBN_to_file()
        for dist in [1,2]:
            pbn_ext = generate_Extended_PBN(bn, i_modifs=None, p_ref=0.8, dist=dist, q=1)
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
test_syntheticPBN(2, 4, 1, indep = False)

# Divers exemples de conversion entre objets PBN et fichiers .pbn
test_filesPBN()

# BN à régulation signée par défaut, et ses extensions en PBN par fonctions voisines
test_extended_PBN()
