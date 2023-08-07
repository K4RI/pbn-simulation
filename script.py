from PBN_simulation import *


def ex_toymodel():
    """Toy model présenté dans l'article d'AbouJaoudé2006."""
    f0 = lambda x: x[1] | x[3]
    f1 = lambda x: (x[0] & x[3]) | x[2]
    f2 = lambda x: (not x[0]) & (not x[3])
    f3 = lambda x: x[3]

    toymodel = PBN(title = 'Toy model',
                   n = 4,
                   indep = False,
                   f = [[f0, f1, f2, f3]],
                   c = [1],
                   sync = True,
                   p = 0,
                   q = 0)
    print(toymodel)
    toymodel.simulation(8, verb = True)
    toymodel.stationary_law()
    toymodel.STG()
    toymodel.copy_PBN(sync = False).STG()



def ex_shmulevich2001():
    """Toy model présenté dans l'article de Shmulevich2001."""

    f0_0 = lambda x: x[1] | x[2]
    f0_1 = lambda x: (x[1] | x[2]) & (not((not x[0]) & x[1] & x[2]))
    f1_0 = lambda x: (x[0] | x[1] | x[2]) & (x[0] | (not x[1]) | (not x[2])) \
                                          & ((not x[0]) | (not x[1]) | x[2])
    f2_0 = lambda x: (x[0] & (x[1] | x[2])) | ((not x[0]) & x[1] & x[2])
    f2_1 = lambda x: x[0] & x[1] & x[2]

    examplePBN = PBN(title = 'Shmulevich PBN',
                     n = 3,
                     indep = True,
                     f = [[f0_0, f0_1], [f1_0], [f2_0, f2_1]],
                     c = [[0.6, 0.4], [1], [0.5, 0.5]],
                     sync = True,
                     p = 0,
                     q = 1)
    print(examplePBN)
    # examplePBN.simulation(20, verb = False)
    examplePBN.STG_PBN()
    examplePBN.stationary_law()



def ex_mammaliancellcycle():
    """Modèle biologique étudié dans Fauré2006."""

    f0 = lambda x: x[0]

    f1 = lambda x: ((not x[4]) & (not x[9]) & (not x[0]) & (not x[3])) \
                 | (x[5] & (not x[9]) & (not x[0]))

    f2 = lambda x: ((not x[1]) & (not x[4]) & (not x[9])) \
                 | (x[5] & (not x[1]) & (not x[9]))

    f3 = lambda x: (x[2] & (not x[1]))

    f4 = lambda x: (x[2] & (not x[1]) & (not x[6]) & (not (x[7] & x[8]))) \
                 | (x[4] & (not x[1]) & (not x[6]) & (not (x[7] & x[8])))

    f5 = lambda x: ((not x[0]) & (not x[3]) & (not x[4]) & (not x[9])) \
                 | (x[5] & (not (x[3] & x[4])) & (not x[9]) &(not x[0]))

    f6 = lambda x: x[9]

    f7 = lambda x: ((not x[4]) & (not x[9])) | (x[6]) | (x[5] & (not x[9]))

    f8 = lambda x: (not x[7]) | (x[7] & x[8] & (x[6] | x[4] | x[9]))

    f9 = lambda x: (not x[6]) & (not x[7])

    cellcycle = PBN(title = 'Mammalian cell cycle',
                    n = 10,
                    indep = False,
                    f = [[f0, f1, f2, f3, f4, f5, f6, f7, f8, f9]],
                    c = [1],
                    sync = True,
                    p = 0,
                    q = 0)
    # cellcycle.simulation(8, verb = True)
    cellcycle.stationary_law(T = 200, N = 500, R = 500, show_all = False)

    prio_attrs = ["1000100000", "1000101111", "1000001010", "1000000010",
                  "1000000110", "1000000100", "1010000100", "1010100100",
                  "1001100100", "1011100000", "1010100000", "1000100100",
                  "1000001110", "1000100011", "1000101011", "1001100000",
                  "1011000100", "1011100100"]

    cellcycle.copy_PBN(sync = False)\
             .stationary_law(T = 200, N = 500, R = 500, show_all = False,
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
                                T = 20, R = 1000)



def test_syntheticBN(n, k):

    gs = [generateBN(n, k, sync = True, v = False, f = False, p = 0),
          generateBN(n, k, sync = True, v = False, f = True, p = 0)]

    for g in gs:
        print(g)
        g.regulation_graph()
        gasync = g.copy_PBN(sync = False)
        if n <= 6: g.STG()
        # g.stationary_law(show_all = False)
        if n <= 6: gasync.STG(layout = nx.spring_layout)
        # gasync.stationary_law(T = 200, N = 500, R = 500, show_all = False)


def test_syntheticPBN(m, n, k, indep):
    gs = [generate_Random_PBN([m,1,m,1], n, k, indep = True, sync = True,
                                                             p = 0, q = .1),
          generate_Random_PBN(m, n, k, indep = indep, sync = True,
                                                      p = 0, q = .1)]

    for g in gs:
        print(g)
        g.STG_PBN()
        g.copy_PBN(sync = False).STG_PBN()
        # g.simulation(50, verb = True)


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
