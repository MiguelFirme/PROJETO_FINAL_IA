import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

HT = ctrl.Antecedent(np.arange(1,5,1), 'Horizonte de Tempo')
TR = ctrl.Antecedent(np.arange(1,5,1), 'Taxa de Risco')
OF = ctrl.Consequent(np.arange(1,5,1), 'Objetivo Financeiro')


Perfil = ctrl.Consequent(np.arange(1,5,1), 'Perfil do Investidor')

HT['Curto'] = fuzz.trimf(HT.universe, [1, 1, 2])
HT['Medio'] = fuzz.trimf(HT.universe, [2, 3, 3])
HT['Longo'] = fuzz.trimf(HT.universe, [3, 4, 4])

TR['Baixo'] = fuzz.trimf(TR.universe, [1, 1, 2])
TR['Medio'] = fuzz.trimf(TR.universe, [2, 3, 3])
TR['Alto'] = fuzz.trimf(TR.universe, [3, 4, 4])

OF['Baixo'] = fuzz.trimf(OF.universe, [1, 1, 2])
OF['Medio'] = fuzz.trimf(OF.universe, [2, 3, 3])
OF['Alto'] = fuzz.trimf(OF.universe, [3, 4, 4])


Perfil['Conservador'] = fuzz.trimf(Perfil.universe, [1, 1, 2])
Perfil['Moderado'] = fuzz.trimf(Perfil.universe, [2, 3, 3])
Perfil['Agressivo'] = fuzz.trimf(Perfil.universe, [3, 4, 4])

rule1 = ctrl.Rule(HT['Curto'] & TR['Baixo'] & OF['Baixo'], Perfil['Conservador'])
rule2 = ctrl.Rule(HT['Medio'] & TR['Medio'] & OF['Medio'], Perfil['Moderado'])
rule3 = ctrl.Rule(HT['Longo'] & TR['Alto'] & OF['Alto'], Perfil['Agressivo'])

perfil_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
perfil_simulador = ctrl.ControlSystemSimulation(perfil_ctrl)

def calcular_perfil_investidor(horizonte_tempo, taxa_risco, objetivo_financeiro):
        perfil_simulador.input['Horizonte de Tempo'] = horizonte_tempo
        perfil_simulador.input['Taxa de Risco'] = taxa_risco
        perfil_simulador.input['Objetivo Financeiro'] = objetivo_financeiro
    
        perfil_simulador.compute()
        resultado = perfil_simulador.output['Perfil do Investidor']
    
        if resultado <= 2:
            return "Conservador"
        elif resultado <= 3:
            return "Moderado"
        else:
            return "Agressivo"


