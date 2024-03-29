from sims_pars.bayesnet import Chromosome
from sims_pars.simulation.actor import FrozenSingleActor, Sampler, CompoundActor


__author__ = 'TimeWz667'


class ParameterCore(Chromosome):
    def __init__(self, nickname, sg, vs, prior):
        Chromosome.__init__(self, vs, prior)
        self.Nickname = nickname
        self.SG = sg
        self.Parent = None
        self.Children = dict()
        self.Actors = None
        self.ChildrenActors = None

    @property
    def Group(self):
        return self.SG.Name

    def breed(self, nickname, group, exo=None):
        """
        Generate an offspring node
        :param nickname: nickname
        :type nickname: str
        :param group: target group of new parameter node
        :type group: str
        :param exo: exogenous variables
        :type exo:
        :return: child parameter core
        """
        if nickname in self.Children:
            raise ValueError('{} has already existed'.format(nickname))
        chd = self.SG.breed(nickname, group, self, exo)
        self.Children[nickname] = chd
        return chd

    def get_sibling(self, nickname, exo=None):
        """
        Generate a sibling node
        :param nickname: nickname
        :type nickname: str
        :param exo: exogenous variables
        :type exo:
        :return: child parameter core
        """
        return self.Parent.breed(nickname, self.Group, exo)

    def get_prototype(self, group, exo=None):
        """
        Generate an offspring node
        :param group: target group of new parameter node
        :type group: str
        :param exo: exogenous variables
        :type exo:
        :return: prototype parameter core
        """
        return self.SG.breed('prototype', group, self, exo)

    def duplicate(self, nickname):
        if not self.Parent:
            raise AttributeError('Root node can not be duplicated')
        return self.Parent.bread(nickname, self.Group, exo=self.Locus)

    def detach_from_parent(self, collect_pars=False):
        """
        Remove the reference to it parent
        :param collect_pars: collect the parameters of the parent to itself
        """
        if not self.Parent:
            return
        self.Parent.remove_children(self.Nickname)
        if collect_pars:
            for k, v in self.Parent:
                self[k] = v
            self.SG.set_local_actors(self)

        self.Parent = None

    def remove_children(self, k):
        """
        Remove a child ParameterCore
        :param k: the name of the child
        :type k: str
        :return: the removed ParameterCore
        """
        try:
            chd = self.Children[k]
            del self.Children[k]
            return chd
        except KeyError:
            pass

    def list_actors(self, shared=True):
        return list(self.get_actors(shared).keys())

    def get_actors(self, shared=True):
        """
        Get all the actors
        :param shared: true for the shared version
        :return: Random variable generators
        :rtype: dict
        """
        if shared and self.Parent:
            try:
                return self.Parent.ChildrenActors[self.Group]
            except (KeyError, TypeError):
                self.SG.put_shared_actors(self)
                return self.Parent.ChildrenActors[self.Group]
        elif self.Actors is None:
            self.SG.put_local_actors(self)
        return self.Actors

    def get_samplers(self, shared=True):
        actors = self.get_actors(shared)
        return {k: Sampler(actor, self) for k, actor in actors.items()}

    def get_sampler(self, sampler, shared=True):
        """
        Get a sampler of a specific variable
        :param sampler: name of the target sampler
        :param shared: true for the shared version
        :return:
        """
        actor = self.get_actors(shared)[sampler]
        return Sampler(actor, self)

    def get_child(self, name):
        return self.Children[name]

    def get_child_samplers(self, group):
        assert group in self.SG.Children

        try:
            ca = self.ChildrenActors[group]
        except KeyError:
            ca = self.SG.SC[group].get_shared_actors(self)
        return ca

    def find_descendant(self, address):
        """
        Find a descendant node
        :param address: str, a series of names of nodes linked with '@'
        :return: a child node in the address
        """
        sel = self
        names = address.split('@')

        for name in names:
            sel = sel.get_child(name)
        return sel

    def impulse(self, imp, bn=None):
        """
        Do interventions
        :param imp: dict(node: value) or list(node), intervention
        :param bn: the original bayesian network
        """
        try:
            g = self.SG.SC.BN.DAG
        except AttributeError:
            g = bn.DAG
        if isinstance(imp, dict):
            shocked = g.downstream(imp.keys())
            non_imp = [k for k, v in imp.items() if v is None]
            imp = {k: v for k, v in imp.items() if v is not None}
            shocked.difference_update(imp.keys())
            shocked = shocked.union(non_imp)
        elif isinstance(imp, list):
            shocked = g.downstream(imp)
            shocked = shocked.union(imp)
            imp = dict()
        else:
            raise AttributeError('imp defined incorrectly')
        # print(shocked)
        self.__set_response(imp, shocked)

    def __set_response(self, imp, shocked):
        shocked_locus = [s for s in shocked if s in self.Locus]
        try:
            shocked_actors = [k for k, v in self.Actors.items() if k in shocked and isinstance(v, FrozenSingleActor)]
        except AttributeError:
            shocked_actors = list()

        shocked_hoist = dict()
        try:
            for k, v in self.ChildrenActors.items():
                try:
                    shocked_hoist[k] = [s for s, t in v.items() if s in shocked and isinstance(t, FrozenSingleActor)]
                except AttributeError:
                    shocked_hoist[k] = dict()
        except AttributeError:
            pass

        self.SG.set_response(imp, shocked_locus, shocked_actors, shocked_hoist, self)

        for v in self.Children.values():
            v.__set_response(imp, shocked)

        if shocked_locus:
            self.reset_probability()

    def __dict__(self):
        return dict(self.Locus)

    def reset_sc(self, sc):
        self.SG = sc[self.SG.Name]
        for v in self.Children.values():
            v.reset_sc(self, sc)

    @property
    def DeepLogPrior(self):
        """
        Log prior with that of offsprings
        :return: log prior probability
        """
        return self.LogPrior + sum(v.DeepLogPrior for v in self.Children.values())

    def __iter__(self):
        if self.Parent:
            for v in iter(self.Parent):
                yield v
        for v in self.Locus.items():
            yield v

    def __getitem__(self, item):
        try:
            return Chromosome.__getitem__(self, item)
        except KeyError:
            try:
                return self.Parent[item]
            except (AttributeError, KeyError, TypeError):
                raise KeyError('{} not found'.format(item))

    def deep_print(self, i=0):
        prefix = '--' * i + ' ' if i else ''
        print('{}{} ({})'.format(prefix, self.Nickname, self))
        for k, chd in self.Children.items():
            chd.deep_print(i + 1)

    def print(self):
        print('{} ({})'.format(self.Nickname, self))

    def clone(self, copy_sc=False, include_children=False):
        if self.Parent:
            raise AttributeError('This is not the root. Please clone from the root node')
        if copy_sc:
            sc = self.SG.SC.clone()
            sg = sc.SGs[self.Group]
        else:
            sg = self.SG
        pc_new = sg.generate(self.Nickname, exo=dict(self))
        pc_new.LogLikelihood = self.LogLikelihood
        pc_new.LogPrior = self.LogPrior

        if include_children:
            self.__children_copy(pc_new)

        return pc_new

    def __children_copy(self, pc_new):
        for k, chd in self.Children.items():
            gp = chd.Group
            chd_new = pc_new.breed(k, gp, exo=chd.Locus)
            chd_new.LogPrior = chd.LogPrior
            chd_new.LogLikelihood = chd.LogLikelihood
            chd.__children_copy(chd_new)


class PseudoParameterCore(ParameterCore):
    def __init__(self, nickname, group="Unknown"):
        ParameterCore.__init__(self, nickname, None, dict(), 0)
        self.__group = group

    @property
    def Group(self):
        return self.__group

    def breed(self, nickname, group, exo=None):
        """
        Generate an offspring node
        :param nickname: nickname
        :type nickname: str
        :param group: target group of new parameter node
        :type group: str
        :param exo: exogenous variables
        :type exo:
        :return: child parameter core
        """
        if nickname in self.Children:
            raise ValueError('{} has already existed'.format(nickname))
        chd = PseudoParameterCore(nickname, group)
        self.Children[nickname] = chd
        return chd

    def get_sibling(self, nickname, exo=None):
        """
        Generate a sibling node
        :param nickname: nickname
        :type nickname: str
        :param exo: exogenous variables
        :type exo:
        :return: child parameter core
        """
        return PseudoParameterCore(nickname, self.Group)

    def get_prototype(self, group, exo=None):
        """
        Generate an offspring node
        :param group: target group of new parameter node
        :type group: str
        :param exo: exogenous variables
        :type exo:
        :return: prototype parameter core
        """
        return PseudoParameterCore('prototype', group)

    def duplicate(self, nickname):
        return PseudoParameterCore(nickname, self.Group)

    def detach_from_parent(self, collect_pars=False):
        """
        Remove the reference to it parent
        :param collect_pars: collect the parameters of the parent to itself
        """
        if not self.Parent:
            return
        self.Parent.remove_children(self.Nickname)
        self.Parent = None

    def remove_children(self, k):
        """
        Remove a child ParameterCore
        :param k: the name of the child
        :type k: str
        :return: the removed ParameterCore
        """
        try:
            chd = self.Children[k]
            del self.Children[k]
            return chd
        except KeyError:
            pass

    def get_samplers(self, include_parent=False):
        """
        Get all the samplers
        :param include_parent: include parent node's samplers or not
        :type include_parent: bool
        :return: Random variable generators
        :rtype: dict
        """
        return list()

    def get_sampler(self, sampler, shared=True):
        """
        Get a sampler of a specific variable
        :param sampler: name of the target sampler
        :param shared: include shared samplers or not
        :return:
        """
        raise KeyError('No sampler in pseudo parameters')

    def get_child(self, name):
        return self.Children[name]

    def get_child_actor(self, group, name):
        try:
            return self.get_child_actors(group)[name]
        except KeyError:
            raise KeyError('Actor not found')

    def get_child_actors(self, group):
        try:
            ca = self.ChildrenActors[group]
        except KeyError:
            ca = self.SG.set_child_actors(self, group)
        return ca

    def find_descendant(self, address):
        """
        Find a descendant node
        :param address: str, a series of names of nodes linked with '@'
        :return: a child node in the address
        """
        sel = self
        names = address.split('@')
        if len(names) < 2:
            return sel
        for name in names[1:]:
            sel = sel.get_child(name)
        return sel

    def impulse(self, imp, bn=None):
        """
        Do interventions
        :param imp: dict(node: value) or list(node), intervention
        :param bn: a bayesian network
        """
        return

    def freeze(self, exo=None):
        exo = exo if exo else dict()
        for k, s in self.get_samplers().items():
            if isinstance(s, CompoundActor):
                self.Locus.update(s.sample_with_mediators(self.Locus))
            else:
                self.Locus[k] = s.render(self.Locus, **exo)

    def __dict__(self):
        return dict()

    def reset_sc(self, sc):
        pass

    @property
    def DeepLogPrior(self):
        """
        Log prior with that of offsprings
        :return: log prior probability
        """
        return 0 + sum(v.DeepLogPrior for v in self.Children.values())

    def __iter__(self):
        if self.Parent:
            for v in iter(self.Parent):
                yield v
        for v in self.Locus.items():
            yield v

    def __getitem__(self, item):
        raise KeyError('{} not found'.format(item))

    def deep_print(self, i=0):
        prefix = '--' * i + ' ' if i else ''
        print('{}{} ({})'.format(prefix, self.Nickname, self))
        for k, chd in self.Children.items():
            chd.deep_print(i + 1)

    def print(self):
        print('{} ({})'.format(self.Nickname, self))

    def clone(self, copy_sc=False, include_children=False):
        if self.Parent:
            raise AttributeError('This is not the root. Please clone from the root node')
        if copy_sc:
            sc = self.SG.SC.clone()
            sg = sc.SGs[self.Group]
        else:
            sg = self.SG
        pc_new = sg.generate(self.Nickname, dict(self))
        pc_new.LogPrior = self.LogPrior

        if include_children:
            self.__children_copy(pc_new)

        return pc_new

    def __children_copy(self, pc_new):
        for k, chd in self.Children.items():
            gp = chd.Group
            chd_new = pc_new.breed(k, gp, exo=chd.Locus)
            chd_new.LogPrior = chd.LogPrior
            chd.__children_copy(chd_new)
