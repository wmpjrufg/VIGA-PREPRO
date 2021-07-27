################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# MATHEUS HENRIQUE MORATO DE MORAES                    ENG. CIVIL / PROF (UFCAT)
# GUSTAVO GONÇALVES COSTA,                                    ENG. CIVIL (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA VIGA PREPRO DESENVOLVIDA PELO GRUPO DE PESQUISA E ESTUDOS EM 
# ENGENHARIA (GPEE)
################################################################################


################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE

def PROP_GEOMETRICA(B_W, H):
    """
    Esta função determina as propriedades geométricas da viga.

    Entrada:
    B_W       | Largura da viga                        | m    | float 
    H         | Altura da viga                         | m    | float

    Saída:
    A_C       | Área da  seção transversal da viga     | m²   | float
    I_C       | Inércia da viga                        | m^4  | float
    Y_SUP     | Ordenada da fibra superior             | m    | float 
    Y_INF     | Ordenada da fibra inferior             | m    | float
    W_SUP     | Modulo de resistência superior         | m³   | float
    W_INF     | Modulo de resistência inferior         | m³   | float
    """
    A_C = B_W * H 
    I_C = (B_W * H ** 3) / 12
    Y_SUP = H / 2 
    Y_INF = H / 2
    W_SUP = I_C / Y_SUP 
    W_INF = I_C / Y_INF 
    return A_C, I_C, Y_SUP, Y_INF, W_SUP, W_INF

def FATOR_BETA1(TEMPO, CIMENTO):
    """
    Esta função calcula o valor de BETA_1 que representa a função de 
    crescimento da resistência do cimento.

    Entrada:
    TEMPO       | Tempo                                          | dias  | float
    CIMENTO     | Cimento utilizado                              |       | string    
                |   'CP1' - Cimento portland 1                   |       | 
                |   'CP2' - Cimento portland 2                   |       |              
                |   'CP3' - Cimento portland 3                   |       |
                |   'CP4' - Cimento portland 4                   |       | 
                |   'CP5' - Cimento portland 5                   |       | 
    
    Saída:
    BETA_1      | Parâmetro de crescimento da resistência        |       | float   
    """
    if TEMPO < 28 :
        if CIMENTO == 'CP1' or CIMENTO == 'CP2':
          S = 0.25  
        elif CIMENTO == 'CP3' or CIMENTO == 'CP4':
          S = 0.38  
        elif CIMENTO == 'CP5':
          S = 0.20  
        BETA_1 = np.exp(S * (1 - (28 / TEMPO) ** 0.50))
    else :
        BETA_1 = 1
    return BETA_1

def MODULO_ELASTICIDADE_CONCRETO(AGREGADO, F_CK, F_CKJ):
    """
    Esta função calcula os módulos de elasticidade do concreto.  

    Entrada:
    AGREGADO    | Tipo de agragado usado no traço do cimento       |        | string    
                |   'BAS' - Agregado de Basalto                    |        | 
                |   'GRA' - Agregado de Granito                    |        |              
                |   'CAL' - Agregado de Calcário                   |        |
                |   'ARE' - Agregado de Arenito                    |        | 
    F_CK        | Resistência característica à compressão          | kN/m²  | float   
    F_CKJ       | Resistência característica à compressão idade J  | kN/m²  | float
    
    Saída:
    E_CIJ       | Módulo de elasticidade tangente                  | kN/m²  | float
    E_CSJ       | Módulo de elasticidade do secante                | kN/m²  | float   
    """
    # Determinação do módulo tangente E_CI idade T
    if AGREGADO == 'BAS':         
        ALFA_E = 1.2
    elif AGREGADO == 'GRA':         
        ALFA_E = 1.0
    elif AGREGADO == 'CAL':       
        ALFA_E = 0.9
    elif AGREGADO == 'ARE':       
        ALFA_E = 0.7
    F_CK /= 1E3
    if F_CK <= 50:        
        E_CI = ALFA_E * 5600 * np.sqrt(F_CK)
    elif F_CK > 50:   
        E_CI = 21.5 * (10 ** 3) * ALFA_E * (F_CK / 10 + 1.25) ** (1 / 3)
    ALFA_I = 0.8 + 0.2 * F_CK / 80
    if ALFA_I > 1:        
        ALFA_I = 1
    # Determinação do módulo secante E_CS idade T
    E_CS = E_CI * ALFA_I
    if F_CK <= 45 :
        F_CK *= 1E3
        E_CIJ = E_CI * (F_CKJ / F_CK) ** 0.5  
    elif  F_CK > 45 : 
        F_CK *= 1E3
        E_CIJ = E_CI * (F_CKJ / F_CK) ** 0.3  
    E_CSJ = E_CIJ * ALFA_I
    E_CIJ *= 1E3 
    E_CSJ *= 1E3 
    return E_CIJ, E_CSJ

def PROP_MATERIAL(F_CK, TEMPO, CIMENTO, AGREGADO):
    """
    Esta função determina propriedades do concreto em uma idade TEMPO.
    
    Entrada:
    F_CK        | Resistência característica à compressão                | kN/m²  | float   
    TEMPO       | Tempo                                                  | dias   | float
    CIMENTO     | Cimento utilizado                                      |        | string    
                |   'CP1' - Cimento portland 1                           |        | 
                |   'CP2' - Cimento portland 2                           |        |              
                |   'CP3' - Cimento portland 3                           |        |
                |   'CP4' - Cimento portland 4                           |        | 
                |   'CP5' - Cimento portland 5                           |        | 
    AGREGADO    | Tipo de agragado usado no traço do cimento             |        | string    
                |   'BAS' - Agregado de Basalto                          |        | 
                |   'GRA' - Agregado de Granito                          |        |              
                |   'CAL' - Agregado de Calcário                         |        |
                |   'ARE' - Agregado de Arenito                          |        | 
    
    Saída:
    F_CKJ       | Resistência característica à compressão idade J        | kN/m²  | float
    F_CTMJ      | Resistência média caracteristica a tração idade J      | kN/m²  | float
    F_CTKINFJ   | Resistência média caracteristica a tração inf idade J  | kN/m²  | float
    F_CTKSUPJ   | Resistência média caracteristica a tração sup idade J  | kN/m²  | float
    E_CIJ       | Módulo de elasticidade tangente                        | kN/m²  | float
    E_CSJ       | Módulo de elasticidade do secante                      | kN/m²  | float      
    """
    # Propriedades em situação de compressão F_C idade TEMPO em dias
    BETA_1 = FATOR_BETA1(TEMPO, CIMENTO)
    F_CKJ = F_CK * BETA_1
    F_CKJ /= 1E3
    F_CK /= 1E3
    if F_CKJ < 21 :
        F_CKJ = 21
    # Propriedades em situação de tração F_CT idade TEMPO em dias
    if F_CK <= 50:
      F_CTMJ = 0.3 * F_CKJ ** (2/3)
    elif F_CK > 50:
      F_CTMJ = 2.12 * np.log(1 + 0.11 * F_CKJ)
    F_CTMJ *= 1E3
    F_CTKINFJ = 0.7 * F_CTMJ 
    F_CTKSUPJ = 1.3 * F_CTMJ
    # Módulo de elasticidade do concreto
    F_CKJ *= 1E3
    F_CK *= 1E3
    [E_CIJ, E_CSJ] = MODULO_ELASTICIDADE_CONCRETO(AGREGADO, F_CK, F_CKJ)
    return  F_CKJ, F_CTMJ, F_CTKINFJ, F_CTKSUPJ, E_CIJ, E_CSJ 

def TENSAO_INICIAL(TIPO_PROT, TIPO_ACO, F_PK, F_YK):
    """
    Esta função determina a tensão inicial de protensão e a carga ini-
    cial de protensão.

    Entrada:
    TIPO_PROT  | Protensão utilizada                                  |       | string    
               |   'PRE' - Peça pré tracionada                        |       | 
               |   'POS' - Peça pós tracionada                        |       |  
    TIPO_ACO   | Tipo de aço                                          |       | string
               |   'RN' - Relaxação normal                            |       |
               |   'RB' - Relaxação baixa                             |       |
    F_PK       | Tensão última característica do aço                  | kN/m² | float
    F_YK       | Tensão de escoamento característica do aço           | kN/m² | float   

    Saída:
    SIGMA_PIT0 | Tensão inicial de protensão                          | kN/m² | float
    
    """
    if TIPO_PROT == 'PRE':
        if TIPO_ACO == 'RN':
            SIGMA_PIT0 = min(0.77 * F_PK, 0.90 * F_YK)
        elif TIPO_ACO == 'RB':
            SIGMA_PIT0 = min(0.77 * F_PK, 0.85 * F_YK)       
    elif TIPO_PROT == 'POS':
        if TIPO_ACO == 'RN':
            SIGMA_PIT0 = min(0.74 * F_PK, 0.87 * F_YK)
        elif TIPO_ACO == 'RB':
            SIGMA_PIT0 = min(0.74 * F_PK, 0.82 * F_YK)
    return SIGMA_PIT0

def COMPRIMENTO_TRANSFERENCIA (PHI_L, F_YK, F_CTKINFJ, ETA_1, ETA_2, SIGMA_PI, H):
    """
    Esta função calcula o comprimento de tranferência da armadura L_P

    Entrada:
    PHI_L      | Diâmetro da armadura                                 | m      | float
    F_YK       | Tensão de escoamento característica do aço           | kN/m²  | float
    F_CTKINFJ  |                                                      |        | float
    ETA_1      |                                                      |        | float
    ETA_2      |                                                      |        | float
    SIGMA_PI   |                                                      |        | float
    H

    Saída:
    L_P 
    """ 
    F_YD = F_YK / 1.15
    F_CTD = F_CTKINFJ / 1.4
    F_BPD = ETA_1 * ETA_2 * F_CTD
    # Comprimento de ancoragem básico para cordoalhas
    L_BP = (7 * PHI_L * F_YD) / (36 * F_BPD)
    # Comprimento básico de transferência para cordoalhas não gradual
    L_BPT = (0.625 * L_BP ) * (SIGMA_PI/ F_YD)
    AUXL_P = np.sqrt(H ** 2 + (0.6 * L_BPT) ** 2) 
    L_P = max(AUXL_P, L_BPT)
    return L_P

def ESFORCOS(Q, L, L_P):
    """
    Esta função determina os esforços atuantes na viga biapoiada.
    
    Entrada:
    Q           | Carga lineramente distribuida      | kN/m    | float
    L           | Comprimento da viga                | m       | float
    L_P

    Saída:
    M          | Momento atuante no meio da viga    | kNm     | float
    M_AP
    V          | Cortante atuante no apoio da viga  | kN      | float
    """
    # Momento no meio do vão
    M_MV = Q * (L ** 2) / 8
    # Momento no apoio nas condições iniciais e finais
    M_AP = (Q * L / 2) * L_P - (Q * L_P / 2) * (L_P / 2)
    # Cortanto nos apoios
    V_AP = Q * L / 2 
    return M_MV, M_AP, V_AP 

def TENSOES_NORMAIS(P_I, A_C, E_P, W_INF, W_SUP, DELTA_P, DELTA_G1, DELTA_G2, DELTA_G3, DELTA_Q1, DELTA_Q2, PSI_Q1, M_G1, M_G2, M_G3, M_Q1, M_Q2):
    """
    Esta função determina a tensão normal nos bordos inferior e superior da peça.
    
    Entrada:
    P_I         | Carga de protensão considerando as perdas         | kN      | float
    A_C         | Área da  seção transversal da viga                | m²      | float
    E_P         | Excentricidade de protensão                       | m       | float 
    W_SUP       | Modulo de resistência superior                    | m³      | float
    W_INF       | Modulo de resistência inferior                    | m³      | float
    DELTA_      | Coeficientes parciais de segurança (G,Q,P)        |         | float
    PSI_Q1      | Coeficiente parcial de segurança carga Q_1        |         | float
    M_          | Momentos caracteristicos da peça (G,Q)            | kNm     | float  
        
    Saída:
    SIGMA_INF   | Tensão normal fibra inferior                      | kN/m²   | float
    SIGMA_SUP   | Tensão normal fibra superior                      | kN/m²   | float
    """
    # Tensão normal fibras inferiores
    # Parcela da protensão
    AUX_PINF =  DELTA_P * (P_I / A_C + P_I * E_P / W_INF) 
    # Parcela da carga permanente de PP
    AUX_G1INF = -1 * DELTA_G1 * M_G1 / W_INF 
    # Parcela da carga permanente da capa
    AUX_G2INF = -1 * DELTA_G2 * M_G2 / W_INF
    # Parcela da carga permanente do revestimento
    AUX_G3INF = -1 * DELTA_G3 * M_G3 / W_INF
    # Parcela da carga acidental de utilização
    AUX_Q1INF = -1 * DELTA_Q1 * PSI_Q1 * M_Q1 / W_INF
    # Parcela da carga acidental de montagem da peça
    AUX_Q2INF = -1 * DELTA_Q2 * M_Q2 / W_INF
    # Total para parte inferior
    SIGMA_INF = AUX_PINF + (AUX_G1INF + AUX_G2INF + AUX_G3INF ) + (AUX_Q1INF + AUX_Q2INF)
    # Tensão normal fibras Superior
    # Parcela da protensão
    AUX_PSUP =  DELTA_P * (P_I / A_C - P_I * E_P / W_SUP) 
    # Parcela da carga permanente de PP
    AUX_G1SUP = 1 * DELTA_G1 * M_G1 / W_SUP 
    # Parcela da carga permanente da capa
    AUX_G2SUP = 1 * DELTA_G2 * M_G2 / W_SUP
    # Parcela da carga permanente do revestimento
    AUX_G3SUP = 1 * DELTA_G3 * M_G3 / W_SUP
    # Parcela da carga acidental de utilização
    AUX_Q1SUP = 1 * DELTA_Q1 * PSI_Q1 * M_Q1 / W_SUP
    # Parcela da carga acidental de montagem da peça
    AUX_Q2SUP = 1 * DELTA_Q2 * M_Q2 / W_SUP
    # Total para parte inferior
    SIGMA_SUP = AUX_PSUP + (AUX_G1SUP + AUX_G2SUP + AUX_G3SUP) + (AUX_Q1SUP + AUX_Q2SUP)
    return SIGMA_INF, SIGMA_SUP

def VERIFICA_TENSAO_NORMAL_ATO_PROTENSÃO(SIGMA_INF, SIGMA_SUP, SIGMA_TRACMAX, SIGMA_COMPMAX):
    """
    Esta função verifica a restrição de tensão normal em peças estruturais conforme
    disposto na seção 17.2.4.3.2 da NBR 6118.
    
    Entrada:
    SIGMA_INF       | Tensão normal fibra inferior                      | kN/m²   | float
    SIGMA_SUP       | Tensão normal fibra superior                      | kN/m²   | float
    SIGMA_TRACMAX   | Tensão normal máxima na tração                    | kN/m²   | float
    SIGMA_COMPMAX   | Tensão normal máxima na compressão                | kN/m²   | float

    Saída:
    G_0             | Valor da restrição análise bordo inferior         |         | float
    G_1             | Valor da restrição análise bordo superior         |         | float
    """
    # Análise bordo inferior
    if SIGMA_INF >= 0:
        SIGMA_MAX = SIGMA_COMPMAX
        SIGMA = SIGMA_INF
    else:
        SIGMA_MAX = SIGMA_TRACMAX
        SIGMA = np.abs(SIGMA_INF)
    G_0 = (SIGMA / SIGMA_MAX) - 1 
    # Análise bordo superior
    if SIGMA_SUP >= 0:
        SIGMA_MAX = SIGMA_COMPMAX
        SIGMA = SIGMA_SUP
    else:
        SIGMA_MAX = SIGMA_TRACMAX
        SIGMA = np.abs(SIGMA_SUP)
    G_1 = (SIGMA / SIGMA_MAX) - 1 
    return G_0, G_1   

def PERDA_DESLIZAMENTO_ANCORAGEM(P_IT0, SIGMA_PIT0, A_SCP, L_0, DELTA_ANC, E_SCP):
    """
    Esta função determina a perda de protensão por deslizamento da armadura na anco-
    ragem.
    
    Entrada:
    P_IT0       | Carga inicial de protensão                        | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                       | kN/m² | float
    A_SCP       | Área de total de armadura protendida              | m²    | float
    L_0         | Comprimento da pista de protensão                 | m     | float
    DELTA_ANC   | Previsão do deslizamento do sistema de ancoragem  | m     | float
    E_SCP       | Módulo de Young do aço protendido                 | kN/m² | float

    Saída:
    DELTAPERC   | Perda percentual de protensão                     | %     | float
    P_IT1       | Carga final de protensão                          | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                       | kN/m² | float
    """
    # Pré-alongamento do cabo
    DELTAL_P = L_0 * (SIGMA_PIT0 / E_P)
    # Redução da deformação na armadura de protensão
    DELTAEPSILON_P = DELTA_ANC / (L_0 +  DELTAL_P)
    # Perdas de protensão
    DELTASIGMA = E_SCP * DELTAEPSILON_P
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def INTERPOLADOR (X_1, X_2, X_K, Y_1, Y_2):
    """
    Esta função interpola lineramente valores.

    Entrada:
    X_1   | Valor inferior X_K     |       | float
    X_2   | Valor superior X_K     |       | float
    Y_1   | Valor inferior Y_K     |       | float
    Y_2   | Valor superior Y_K     |       | float
    X_K   | Valor X de referência  |       | float

    Saída:
    Y_K   | Valor interpolado Y    |       | float
    """
    Y_K = Y_1 + (X_K - X_1) * ((Y_2 - Y_1) / (X_2 - X_1))
    return Y_K 

def TABELA_PSI1000(TIPO_FIO_CORD_BAR, TIPO_ACO, RHO_SIGMA):
    """
    Esta função encontra o fator Psi 1000 para cálculo da relaxação.

    Entrada:
    TIPO_FIO_CORD_BAR  | Tipo de armadura de protensão de acordo com a aderência escolhida                 |       | string
                       |    'FIO' - Fio                                                                    |       |
                       |    'COR' - Cordoalha                                                              |       |
                       |    'BAR' - BARRA                                                                  |       |
    TIPO_ACO           | Tipo de aço                                                                       |       | string
                       |    'RN' - Relaxação normal                                                        |       |
                       |    'RB' - Relaxação baixa                                                         |       |
    RHO_SIGMA          | Razão entre F_PK e SIGMA_PI                                                       |       | float

    Saída:
    PSI_1000           | Valor médio da relaxação, medidos após 1.000 h, à temperatura constante de 20 °C  | %     | float     
    """
    # Cordoalhas
    if TIPO_FIO_CORD_BAR == 'COR':
        if TIPO_ACO == 'RN':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 3.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 3.50; Y_1 = 7.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 7.00; Y_1 = 12.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA 
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)   
        elif TIPO_ACO == 'RB':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.30
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.30; Y_1 = 2.50
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 2.50; Y_1 = 3.50
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
    # Fio
    elif TIPO_FIO_CORD_BAR == 'FIO':
        if TIPO_ACO == 'RN':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 2.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 2.50; Y_1 = 5.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 5.00; Y_1 = 8.50
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA   
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1) 
        elif TIPO_ACO == 'RB':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0 
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.00
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.00; Y_1 = 2.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 2.00; Y_1 = 3.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA  
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)  
    # Barra
    elif TIPO_FIO_CORD_BAR == 'BAR':
        if RHO_SIGMA <= 0.5:
                PSI_1000 = 0 
        elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1) 
        elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.50; Y_1 = 4.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1) 
        elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 4.00; Y_1 = 7.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA 
                PSI_1000 = INTERPOLADOR (X_0, X_1, X_K, Y_0, Y_1)        
    return PSI_1000       

def PERDA_RELAXACAO_ARMADURA(P_IT0, SIGMA_PIT0, T_0, T_1, TEMP, F_PK, A_SCP, TIPO_FIO_CORD_BAR, TIPO_ACO):
    """
    Esta função determina a perda de protensão por relaxação da armadura de protensão
    em peças de concreto protendido. 
    
    Entrada:
    P_IT0              | Carga inicial de protensão                                         | kN    | float
    SIGMA_PIT0         | Tensão inicial de protensão                                        | kN/m² | float
    T_0                | Tempo inicial de análise sem correção da temperatura               | dias  | float
    T_1                | Tempo final de análise sem correção da temperatura                 | dias  | float 
    TEMP               | Temperatura de projeto                                             | °C    | float 
    F_PK               | Tensão última do aço                                               | kN/m² | float
    A_SCP              | Área de total de armadura protendida                               | m²    | float
    TIPO_FIO_CORD_BAR  | Tipo de armadura de protensão de acordo com a aderência escolhida  |       | string
                       |    'FIO' - Fio                                                     |       |
                       |    'COR' - Cordoalha                                               |       |
                       |    'BAR' - BARRA                                                   |       |
    TIPO_ACO           | Tipo de aço                                                        |       | string
                       |    'RN' - Relaxação normal                                         |       |
                       |    'RB' - Relaxação baixa                                          |       |
      
    Saída:
    DELTAPERC          | Perda percentual de protensão                                      | %     | float
    P_IT1              | Carga final de protensão                                           | kN    | float
    SIGMA_PIT1         | Tensão inicial de protensão                                        | kN/m² | float
    """
    # Determinação PSI_1000
    RHO_SIGMA = SIGMA_PIT0 / F_PK 
    if T_1 > (20 * 365):  
          PSI_1000 = 2.5
    else:
          PSI_1000 = TABELA_PSI1000(TIPO_FIO_CORD_BAR, TIPO_ACO, RHO_SIGMA)
    # Determinação do PSI no intervalo de tempo T_1 - T_0
    DELTAT_COR = (T_1 - T_0) * TEMP / 20
    PSI =  PSI_1000 * (DELTAT_COR / 41.67) ** 0.15
    # Perdas de protensão
    DELTASIGMA = PSI * SIGMA_PIT0
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def PERDA_DEFORMACAO_CONCRETO(E_SCP, E_CCP, P_IT0, SIGMA_PIT0, A_C, I_C, E_P, M_GPP):
    """
    Esta função determina a perda de protensão devido a deformação inicial do concreto. 
    
    Entrada:
    E_SCP       | Módulo de Young do aço protendido                 | kN/m² | float
    E_CCP       | Módulo de Young do concreto                       | kN/m² | float
    P_IT0       | Carga inicial de protensão                        | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                       | kN/m² | float
    A_C         | Área bruta da seção                               | m²    | float 
    I_C         | Inércia da seção bruta                            | m^4   | float
    E_P         | Excentricidade de protensão                       | m     | float 
    M_GPP       | Momento fletor devido ao peso próprio             | kN.m  | float 
      
    Saída:
    DELTAPERC   | Perda percentual de protensão                     | %     | float
    P_IT1       | Carga final de protensão                          | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                       | kN/m² | float
    """
    # Perdas de protensão
    ALPHA_P = E_SCP / E_CCP
    AUX_0 = P_IT0 / A_C
    AUX_1 = (P_IT0 * E_P ** 2) / I_C
    AUX_2 = (M_GPP * E_P) / I_C
    DELTASIGMA = ALPHA_P * (AUX_0 + AUX_1 - AUX_2)
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def CALCULO_HFIC(U, A_C, MU_AR):
    """
    Esta função calcula a altura fictícia de uma peça de concreto.

    Entrada:
    U       | Umidade do ambiente no intervalo de tempo de análise         | %     | float
    A_C     | Área bruta da seção                                          | m²    | float
    MU_AR   | Parte do perímetro externo da seção em contato com ar        | m     | float

    Saída:
    H_FIC   | Altura fictícia da peça para cálculo de fluência e retração  | m     | float
    """
    GAMMA = 1 + np.exp(-7.8 + 0.1 * U)
    H_FIC = GAMMA * 2 * A_C / MU_AR
    if H_FIC > 1.60:
        H_FIC = 1.60
    if H_FIC < 0.050:
        H_FIC = 0.050
    return H_FIC

def CALCULO_TEMPO_FICTICIO(T, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO):
    """
    Esta função calcula o tempo corrigido para cálculo das perdas de fluência e retração. 

    Entrada:
    T                   | Tempo para análise da correção em função da temperatura    | dias  | float
    TEMP                | Temperatura de projeto                                     | °C    | float 
    TIPO_PERDA          | Tipo da perda que deseja-se calcular a correção do tempo   |       | string
                        |       'LENTO'  - Endurecimento lento AF250, AF320, POZ250  |       |
                        |       'NORMAL' - Endurecimento normal CP250, CP320, CP400  |       |
                        |       'RAPIDO' - Endurecimento rápido aderência            |       |
    TIPO_ENDURECIMENTO  | Tipo de enduricmento do cimento                            |       | string
                        |       'RETRACAO' - Retração                                |       |
                        |       'FLUENCIA' - Fluência                                |       |                                                                                           

    Saída:
    T_COR               | Tempo de projeto corrigido em função da temperatura        | °C    | float 
    """
    # Parâmetros de reologia e tipo de pega
    if TIPO_PERDA == 'RETRACAO':
        ALFA = 1
    elif TIPO_PERDA == 'FLUENCIA':
        if TIPO_ENDURECIMENTO == 'LENTO':
            ALFA = 1
        elif TIPO_ENDURECIMENTO == 'NORMAL':
            ALFA = 2
        elif TIPO_ENDURECIMENTO == 'RAPIDO':
            ALFA = 3
    # Correção dos tempos menores que 3 dias e maiores que 10.000 dias
    if T < 3 and T > 0:
        T = 3
    elif T > 10000:
        T = 10000
    # Determinação da idade fictícia do concreto
    T_COR = ALFA * ((TEMP + 10) / 30) * T
    return T_COR 

def PERDA_RETRACAO_CONCRETO(U, ABAT):
    """
    Esta função determina a perda de protensão devido a retração do concreto. 
    
    Entrada:
    U           | Umidade do ambiente no intervalo de tempo de análise   | %     | float
    ABAT        | Abatimento ou slump test do concreto                   | kN/m² | float
    A_C         | Área bruta da seção                                    | m²    | float 
    MU_AR       | Parte do perímetro externo da seção em contato com ar  | m     | float


    I_C         | Inércia da seção bruta                                 | m^4   | float
    E_P         | Excentricidade de protensão                            | m     | float 
    M_GPP       | Momento fletor devido ao peso próprio                  | kN.m  | float 
    P_IT0       | Carga inicial de protensão                             | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                            | kN/m² | float
      
    Saída:
    DELTAPERC   | Perda percentual de protensão                         | %     | float
    P_IT1       | Carga final de protensão                              | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                           | kN/m² | float
    """
    # Cálculo da defomração específica EPSILON_1S
    EPSILON_1S = -8.09 + (U / 15) - (U ** 2 / 2284) - (U ** 3 / 133765) + (U ** 4 / 7608150)
    EPSILON_1S /= 1E4 
    if U <= 90 and (ABAT >= 5 and ABAT <= 9):          # intervalo 0.05 <= ABAT <= 0.09
        EPSILON_1S = EPSILON_1S
    elif U <= 90 and (ABAT >= 0 and ABAT <= 4):        # intervalo 0.00 <= ABAT <= 0.04
        EPSILON_1S *= 0.75
    elif U <= 90 and (ABAT >= 10 and ABAT <= 15):      # intervalo 10.0 <= ABAT <= 15.0
        EPSILON_1S *= 1.25
    # Cálculo da defomração específica EPSILON_2S
    H_FIC = CALCULO_HFIC(U, A_C, MU_AR)
    H_FIC *= 100
    EPSILON_2S = (33 + 2 * H_FIC) / (20.8 + 3 * H_FIC)
    # Valor final da deformação por retração
    EPSILON_CS = EPSILON_1S * EPSILON_2S
    

    DELTASIGMA = ALPHA_P * (AUX_0 + AUX_1 - AUX_2)
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def VERIFICACAO_CISALHAMENTO(F_CK, B_W, D, A_SW, F_YK, TIPO_VC, M_SDMAX, P_I, A, W, E_P):
    """
    Esta função verifica o valor dos parâmetors V_R2, V_SW e V_C para uma peça
    de concreto armado / protendido.

    Entrada:
    F_CK        | Resistência característica à compressão         | MPa   | float
    F_YK        | Resistência característica do aço               | MPa   | float
    B_W         | Largura da viga                                 | cm    | float  
    D           | Altura útil da seção                            | cm    | float
    A_SW        | Área de aço para cisalhamento                   | cm²/m | float
    TIPO_VC     | #############                                   |       | string
                |       'TRACAO'                                  |       |
                |       'FS'                                      |       |
                |       'FC'                                      |       |
    M_SDMAX     | Momento de cálculo máximo                       | kN.cm | float
    P_I         | Carga de protensão considerando as perdas       | kN    | float
    A           | Área da seção transversal da viga               | cm²   | flot
    E_P         | Excentricidade de protensão                     | cm²   | float 
    W           | Modulo de resistência superior                  | cm³   | float

    Saída:
    V_RD2       | Resitência da biela comprimida                  | kN    | float 
    V_SW        | Resitência ao cisalhamento da armadura          | kN    | float
    V_C         | Resitência ao cisalhamento do concreto          | kN    | float
    """
    # Força resistente da biela de compressão
    ALFA_V2 = (1 - (F_CK / 250))
    F_CD = F_CK / 1.40
    F_CD = F_CD / 10
    V_RD2 = 0.27 * ALFA_V2 * F_CD * B_W * D
    # Força resistente da armadura de cisalhamento
    F_YWD = (F_YK / 1.15) / 10
    V_SW = A_SW * 0.9 * F_YWD * (np.sin (np.pi / 2) + np.cos (np.pi / 2))
    # Força resistente do concreto
    if F_CK <= 50:
        F_CTM = 0.3 * F_CK ** (2 / 3)
    else:
        F_CTM = 2.12 * np.log(1 + 0.11 * F_CK)
    F_CTKINF = 0.70 * F_CTM
    F_CTD = F_CTKINF / 1.4 
    F_CTD = F_CTD / 10
    if TIPO_VC == 'TRACAO':
        V_C = 0
    elif TIPO_VC == 'FS':
        V_C0 = 0.6 * F_CTD * B_W * D
        V_C = V_C0
    elif TIPO_VC == 'FC':
        V_C0 = 0.6 * F_CTD * B_W * D
        SIGMA = P_I / A + (P_I * E_P) / W
        M_0 = 0.90 * W * SIGMA
        V_CCALC = V_C0 * (1 + M_0 / M_SDMAX)
        if V_CCALC > 2 * V_C0:
            V_C = 2 * V_C0
        else:
            V_C = V_CCALC
    return V_RD2, V_SW, V_C 

def TENSAO_ACO(E_SCP, EPSILON, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a tensão da armadura de protensão a partir de 
    um valor de deformação.

    Entrada:
    E_SCP       | Módulo de elasticidade do aço protendido        | MPa   | float
    EPSILON     | Deformação correspondente a tensão SIGMA 
    desejada                                                      |       | float
    EPSILON_P   | Deformação última do aço                        |       | float
    EPSILON_Y   | Deformação escoamento do aço                    |       | float
    F_Y         | Tensão de escoamento do aço                     | MPa   | float
    F_P         | Tensão última do aço                            | MPa   | float
    
    Saída:
    SIGMA       | Tensão correspondente a deformação Deformação   | kN/cm²| float
    """
    # Determinação da tensão SIGMA correspodente a deformação EPSILON
    if EPSILON < (F_Y / E_SCP) :
        SIGMA = E_SCP * EPSILON
    elif EPSILON >= (F_Y / E_SCP):
        AUX = (F_P - F_Y) / (EPSILON_P - EPSILON_Y)
        SIGMA = F_Y + AUX * (EPSILON - EPSILON_Y)
    SIGMA = SIGMA / 10
    return SIGMA

def DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a deformação da armadura de protensão a partir de 
    um valor de tensão

    Entrada:
    E_SCP       | Módulo de elasticidade do aço protendido        | MPa   | float
    SIGMA       | Tensão correspondente a tensão EPSILON desejada | kN/cm²| float
    EPSILON_P   | Deformação última do aço                        |       | float
    EPSILON_Y   | Deformação escoamento do aço                    |       | float
    F_Y         | Tensão de escoamento do aço                     | MPa   | float
    F_P         | Tensão última do aço                            | MPa   | float
    
    Saída:
    EPSILON     | Deformação correspondente a tensão SIGMA 
    desejada                                                      |       | float
    """
    # Determinação da deformação EPSILON correspodente a tensão SIGMA
    SIGMA = SIGMA * 10
    if (SIGMA / E_SCP) < (F_Y / E_SCP):      
        EPSILON = SIGMA / E_SCP
    elif (SIGMA / E_SCP) >= (F_Y / E_SCP):
        AUX = (F_P - F_Y) / (EPSILON_P - EPSILON_Y)
        EPSILON = (SIGMA - F_Y) / AUX + EPSILON_Y
    return EPSILON

def AREA_ACO_FNS_RETANGULAR_SIMPLES(TIPO_CONCRETO, M_D, F_CK, B_W, D, E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a área de aço em elementos de concreto quando submeti-
    do a um momento fletor M_SD
    
    TIPO_CONCRETO|                                                 |       |string
                 |       'CP' - Concreto protendido                |       |
                 |       'CA' - Concreto armado                    |       |
    M_D          | Momento de cálculo                              | kN.cm | float
    F_CK         | Resistência característica à compressão         | MPa   | float
    B_W          | Largura da viga                                 | cm    | float  
    D            | Altura útil da seção                            | cm    | float
    E_SCP        | Módulo de elasticidade do aço protendido        | MPa   | float
    SIGMA        | Tensão correspondente a tensão EPSILON desejada | kN/cm²| float
    EPSILON_P    | Deformação última do aço                        |       | float
    EPSILON_Y    | Deformação escoamento do aço                    |       | float
    F_Y          | Tensão de escoamento do aço                     | MPa   | float
    F_P          | Tensão última do aço                            | MPa   | float

    Saída:
    X            | Linha neutra da seção medida da parte externa
                   comprimida ao CG                                | cm    | float  
    Z            | Braço de alvanca                                | cm    | float    
    A_S          | Área de aço necessária na seção                 | cm²   | float
    EPSILON_S    | Deformação do aço                               |       | float
    EPSILON_C    | Deformação do concreto                          |       | float
    """
    # Determinação dos fatores de cálculo de X e A_S
    if F_CK >  50:
        LAMBDA = 0.80 - ((F_CK - 50) / 400)
        ALPHA_C = (1.00 - ((F_CK - 50) / 200)) * 0.85
        EPSILON_C2 = 2 + 0.085 * (F_CK - 50) ** 0.53
        EPSILON_C2 = EPSILON_C2 / 1000
        EPSILON_CU = 2.6 + 35 * ((90 - F_CK)/100) ** 4
        EPSILON_CU = EPSILON_CU / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.35
    else:
        LAMBDA = 0.80
        ALPHA_C = 0.85
        EPSILON_C2 = 2 / 1000
        EPSILON_CU = 3.5 / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.35
    # Linhas neutra X
    F_CD = F_CK / 1.40
    F_CD = F_CD / 10
    PARTE_1 = M_D / (B_W * ALPHA_C * F_CD)
    NUMERADOR = D - np.sqrt(D ** 2 - 2 * PARTE_1)
    DENOMINADOR = LAMBDA
    X = NUMERADOR / DENOMINADOR
    # Deformações nas fibras comprimidas (concreto) e tracionadas (aço) 
    KX = X / D
    if KX > KX_23:
        EPSILON_C = EPSILON_CU
        EPSILON_S = (1 -  KX) * EPSILON_C 
    elif KX < KX_23:
        EPSILON_S = 10 / 1000
        EPSILON_C = EPSILON_S / (1 - KX)
    elif KX == KX_23:
        EPSILON_S = 10 / 1000
        EPSILON_C = EPSILON_CU 
    # Braço de alavanca Z
    Z = D - 0.50 * LAMBDA * X
    # Área de aço As
    if TIPO_CONCRETO == 'CP':
        EPSILON_SAUX = DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y)
        EPSILON_ST = EPSILON_S + EPSILON_SAUX
        F_YD = TENSAO_ACO(E_SCP, EPSILON_ST, EPSILON_P, EPSILON_Y, F_P, F_Y)
    elif TIPO_CONCRETO == 'CA':
        F_YD = F_Y / 1.15
        F_YD = F_YD / 10
    A_S = M_D / (Z * F_YD)
    return X, EPSILON_S, EPSILON_C, Z, A_S

def MOMENTO_MINIMO(W_0, F_CK):
    """
    Esta função calcula o momento mínimo para a área de aço mínima 

    Entrada:
    W_0          | Módulo de resistência da seção transversal 
    bruta de concreto, relativo à fibra mais tracionada            | cm³   | float
    F_CK         | Resistência característica à compressão         | MPa   | float

    Saída:
    M_MIN        | Momento mínimo para armadura mínima             | kN.cm | float
    """
    # Resistência à tração do concreto
    if F_CK <= 50:
        F_CTM = 0.3 * F_CK ** (2 / 3)
    else:
        F_CTM = 2.12 * np.log(1 + 0.11 * F_CK)
    F_CTKSUP = 1.3 * F_CTM
    F_CTKSUP = F_CTKSUP / 10
    # Momento mínimo
    M_MIN = 0.80 * W_0 * F_CTKSUP
    return M_MIN

def ARMADURA_ASCP_ELS(A_C, W_INF, E_P, PSI1_Q1, PSI2_Q1, M_G1, M_G2, M_G3, M_Q1, SIGMA_PI, F_CTKINFJ):
    """
    Esta função calcula a área de aço mínima em função dos limites do ELS.

    Entrada:
    A_C         | Área da  seção transversal da viga                       | m²      | float
    W_INF       | Modulo de resistência inferior                           | m³      | float
    E_P         | Excentricidade de protensão                              | m       | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                   |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                   |         | float
    M_          | Momentos caracteristicos da peça (G,Q)                   | kN.m    | float
    SIGMA_PI    | Tensão de protensão                                      | kN/m²   | float
    F_CTKINFJ   | Resistência caracteristica a tração inferior na idade j  | kN/m²   | float 

    Saída:
    A_SCPINICIAL| Área de aço inicial respeitando os limites de serviço    | m²      | float
    """
    LIMITE_TRAC0 = -1.50 * F_CTKINFJ
    AUX_0 = SIGMA_PI / A_C + (SIGMA_PI * E_P) / W_INF
    AUX_1 = (M_G1 + M_G2 + M_G3) / W_INF + (PSI1_Q1 * M_Q1) / W_INF
    A_SCP0 = (LIMITE_TRAC0 +  AUX_1) / AUX_0
    LIMITE_TRAC1 = 0
    AUX_2 = (M_G1 + M_G2 + M_G3) / W_INF + (PSI2_Q1 * M_Q1) / W_INF
    A_SCP1 = (LIMITE_TRAC1 + AUX_2) / AUX_0
    A_SCPINICIAL = max(A_SCP0, A_SCP1)
    return A_SCPINICIAL

def ARMADURA_ASCP_ELU(A_C, W_INF, W_SUP, E_P, PSI1_Q1, PSI2_Q1, M_G1, M_G2, M_G3, M_Q1, SIGMA_PI, F_CTMJ, F_CKJ):
    """
    Esta função calcula a área de aço mínima em função dos limites do ELU.

    Entrada:
    A_C         | Área da  seção transversal da viga                       | m²      | float
    W_INF       | Modulo de resistência inferior                           | m³      | float
    W_SUP       | Modulo de resistência superior                           | m³      | float
    E_P         | Excentricidade de protensão                              | m       | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                   |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                   |         | float
    M_          | Momentos caracteristicos da peça (G,Q)                   | kN.m    | float
    SIGMA_PI    | Tensão de protensão                                      | kN/m²   | float
    F_CTMJ      | Resistência caracteristica a tração média na idade j     | kN/m²   | float
    F_CKJ       | Resistência característica à compressão idade j          | kN/m²  | float 

    Saída:
    A_SCPINICIAL| Área de aço inicial respeitando os limites do ato de     | m²      | float
                  protensão  
    """
    LIMITE_TRAC0 = -1.20 * F_CTMJ
    AUX_0 = SIGMA_PI / A_C - (SIGMA_PI * E_P) / W_SUP
    AUX_1 = (M_G1 + M_G2 + M_G3) / W_SUP + (PSI1_Q1 * M_Q1) / W_SUP
    A_SCP0 = (LIMITE_TRAC0 -  AUX_1) / AUX_0
    LIMITE_COMP0 = 0.70 * F_CKJ
    AUX_2 = (M_G1 + M_G2 + M_G3) / W_INF + (PSI2_Q1 * M_Q1) / W_INF
    AUX_3 = SIGMA_PI / A_C + (SIGMA_PI * E_P) / W_INF
    A_SCP1 = (LIMITE_COMP0 + AUX_2) / AUX_3
    return A_SCP0, A_SCP1

def ABERTURA_FISSURAS(ALFA_E, P_IINF, A_2, M_SDMAX, D, X_2, I_2, DIAMETRO_ARMADURA, ETA_COEFICIENTE_ADERENCIA, E_SCP, F_CTM, RHO_R) :
    """
    Esta função calcula a abertura de fissuras na peça 

    Entrada:
    ALFA_E      | Relação dos modulos                                    | m²      | float
    W_INF       | Modulo de resistência inferior                         | m³      | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                 |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                 |         | float
    """   
    SIGMA_S = ALFA_E * (P_IINF / A_2) + ( ALFA_E * (M_SDMAX * (D - X_2) / I_2 ) )
    W_1 = (DIAMETRO_ARMADURA / 12.5 * ETA_COEFICIENTE_ADERENCIA) * (SIGMA_S / E_SCP) * 3 (SIGMA_S / F_CTM)
    W_2 = (DIAMETRO_ARMADURA / 12.5 * ETA_COEFICIENTE_ADERENCIA) * (SIGMA_S / E_SCP) * ((4 / RHO_R) + 45)
    W_FISSURA = min(W_1, W_2)
    return W_FISSURA

