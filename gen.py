import numpy as np

a = ""
for i in range(8):
    for j in range(16):
        index = i + 8 * j
        s = "i_ib_data[" + str(index) + "], "
        a = s + a
#print(a)


class mul():
    def __init__(self,
                 a,
                 b):
        self.a = a
        self.b = b

    def op(self):
        return self.a * self.b


aa = mul(1, 2)
bb = aa.op()
print(bb)

