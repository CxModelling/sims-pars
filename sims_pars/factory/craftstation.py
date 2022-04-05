from pydantic import BaseModel, ValidationError
from abc import ABCMeta, abstractmethod
import ast

__author__ = 'Chu-Chang Ku'
__all__ = ['Atelier', 'AbsCreator']


class AbsCreator(BaseModel, metaclass=ABCMeta):
    @abstractmethod
    def create(self):
        pass


class Atelier:
    def __init__(self):
        self.Creators = dict()

    def register(self, name, creator):
        self.Creators[name] = creator

    def complete_def(self, seq):
        root = ast.parse(seq)
        for inputs in ast.walk(root):
            if isinstance(inputs, ast.Call):
                break
        else:
            raise AttributeError('Unknown input type')

        tp = inputs.func.id
        req = self.Creators[tp].schema()['properties']

        args = [arg.value if isinstance(arg, ast.Constant) else ast.unparse(arg) for arg in inputs.args]

        named = [arg.arg for arg in inputs.keywords]
        unnamed = {k: v for k, v in req.items() if k not in named}

        if len(unnamed) == 0:
            return seq

        inputs.args = list()

        full = ast.unparse(inputs)

        temp = list()
        for i, (k, v) in enumerate(unnamed.items()):
            try:
                v = args[i]
            except IndexError:
                v = v['default']
            temp.append(f'{k}={v}')

        unnamed = f'({", ".join(temp)}'

        full = full.replace('(', unnamed + (', ' if len(named) > 0 else ''), 1)
        return full

    def get_blueprint(self, seq, loc=None, to_complete=True):
        if to_complete:
            seq = self.complete_def(seq)
        return eval(seq, self.Creators, loc)

    def create(self, seq, loc=None, append_src=False, to_complete=True):
        bp = self.get_blueprint(seq, loc=loc, to_complete=to_complete)
        obj = bp.create()

        if append_src:
            try:
                obj.source = seq
            except AttributeError:
                pass
            try:
                obj.json = bp.dict()
            except AttributeError:
                pass

        return obj

    def get_schema(self, tp):
        return self.Creators[tp].schema()

    def list(self):
        return list(self.Creators.keys())

    def __contains__(self, item):
        return item in self.Creators

    def __str__(self):
        products = list(self.Creators.keys())
        if products:
            return 'The workshop of ' + ', '.join(products)
        else:
            return 'A new workshop'


if __name__ == '__main__':
    class A(AbsCreator):
        scale: float = 1

        def create(self):
            return f'A scale={self.scale}'


    ate = Atelier()
    ate.register('A', A)

    print(ate.complete_def('A()'))
    print(ate.complete_def('A(5)'))

    a = ate.create('A()')
    print(a)

    a = ate.create('A(scale=k)', {'k': 5})
    print(a)

