import numpy as np
from sims_pars.fitting.base import AbsObjectiveSC, AbsObjective
from sims_pars.bayesnet import Chromosome
from sims_pars.simulation import get_all_fixed_sc
from joblib import Parallel, delayed


__all__ = ['draw', 'mutate_and_draw', 'mutate_and_draw_parallel', 'AbsObjectiveSC']


def draw(obj: AbsObjective, unpack=False):
    p, li, i = None, np.inf, 0
    while np.isinf(li):
        p = obj.sample_prior()
        li = obj.evaluate(p)
        i += 1
        if i > 20:
            p = Chromosome()
            p.LogLikelihood = - np.inf
            break
    if unpack:
        return p.to_json(), i
    else:
        return p, i


def mutate(p0: Chromosome, sizes):
    p = p0.clone()
    p.reset_probability()
    if isinstance(sizes, dict):
        for k, v in sizes.items():
            p.Locus[k] += v
    elif isinstance(sizes, float):
        for k in p0.Locus.keys():
            p.Locus[k] += sizes
    else:
        raise TypeError('Unknown types of sizes')
    return p


def mutate_and_draw(obj: AbsObjective, p0: Chromosome, scale, unpack=False):
    p, li, i = p0, np.inf, 0
    while np.isinf(li):
        sizes = {k: np.random.normal(0, v) for k, v in scale.items()}
        p = mutate(p0, sizes)

        obj.evaluate_prior(p)
        if np.isinf(p.LogPrior):
            continue
        li = obj.evaluate(p)
        i += 1

        if i > 20:
            p = Chromosome()
            p.LogLikelihood = - np.inf
            break

    if unpack:
        return p.to_json(), i
    else:
        return p, i


def __mutate_and_draw(obj: AbsObjective, p0: dict, scale: dict):
    p, li, i = p0, np.inf, 0
    while np.isinf(li):
        sizes = {k: np.random.normal(0, v) for k, v in scale.items()}

        p = {k: p0[k] + v for k, v in sizes.items()}
        try:
            p = obj.serve(p)
        except ValueError:
            continue
        obj.evaluate_prior(p)
        if np.isinf(p.LogPrior):
            continue
        li = obj.evaluate(p)
        i += 1

        if i > 20:
            p = Chromosome()
            p.LogLikelihood = - np.inf
            break

    return p.to_json(), i


def mutate_and_draw_parallel(obj: AbsObjective, p0s, scale, parallel: Parallel):
    p0s_loc = [p0.Locus for p0 in p0s]

    with parallel:
        ps = parallel(delayed(__mutate_and_draw)(obj, p0, scale) for p0 in p0s_loc)
    return [(obj.serve_from_json(p), i) for p, i in ps]


if __name__ == '__main__':
    from joblib import Parallel, delayed

    class BetaBinSC(AbsObjectiveSC):
        def simulate(self, pars):
            sim = {
                'x1': pars['x1'],
                'x2': pars['x2']
            }
            return sim

        def link_likelihood(self, sim):
            return -((sim['x1'] - 5) ** 2 + (sim['x2'] - 10) ** 2)


    scr = '''
    PCore BetaBin {
        al = 1
        be = 1
    
        p1 ~ beta(al, be)
        p2 ~ beta(al, be)
    
        x1 ~ binom(10, p1)
        x2 ~ binom(n2, p2) 
    }
    '''

    model0 = BetaBinSC(get_all_fixed_sc(scr), exo={'n2': 20})
    model0.print()

    p1, _ = draw(model0)
    print(p1)

    si = {'p1': 0.1, 'p2': 0.5}

    print(mutate(p1, si))

    print(mutate(p1, si))

    print('Mutate draw')
    print(mutate_and_draw(model0, p1, si)[0])
    print(mutate_and_draw(model0, p1, si)[0])
    print(mutate_and_draw(model0, p1, si)[0])

    ps = mutate_and_draw_parallel(model0, [p1, p1, p1], si, Parallel(n_jobs=4, verbose=6))
    for p in ps:
        print(p)
