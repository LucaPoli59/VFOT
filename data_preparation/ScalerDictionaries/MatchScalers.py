"""

File d'incapsulamento per il ritorno dello scaler corretto

"""

from main_dir import constant as const
from data_preparation.ScalerDictionaries import ScalerDictionaryBM, ScalerDictionaryQBM, ScalerDictionaryM, \
    ScalerDictionaryRM, ScalerDictionaryR


def match_scalers():
    """
    Funzione che restituisce lo scaler del tipo definito nel file constat
    :return: scaler del tipo corretto
    """
    match const.SCALER_TYPE:
        case "QBM":
            return ScalerDictionaryQBM.ScalerDictionaryQBM()
        case "BM":
            return ScalerDictionaryBM.ScalerDictionaryBM()
        case "R":
            return ScalerDictionaryR.ScalerDictionaryR()
        case "M":
            return ScalerDictionaryM.ScalerDictionaryM()
        case "RM":
            return ScalerDictionaryRM.ScalerDictionaryRM()
        case _:
            return None
