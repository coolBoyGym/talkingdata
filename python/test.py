class B:
    def apply(self, **argv2):
        pass


class A:
    def __init__(self, B):
        self.B = B

    def get_b(self, **argv2):
        return self.B.apply(**argv2)


class C(B):
    def apply(self, i, j, k):
        print i
        print j
        print k

a = A(C())
a.get_b(i=1, j=2, k=3)
