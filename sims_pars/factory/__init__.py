from sims_pars.factory.craftstation import Atelier, AbsCreator

__all__ = ['get_atelier', 'AbsCreator']


AtelierDict = dict()


def get_atelier(name):
    assert isinstance(name, str)

    try:
        ws = AtelierDict[name]
    except KeyError:
        AtelierDict[name] = ws = Atelier()

    return ws

