import numpy as np
x_input = np.random.rand(128, 24, 24, 3)
y_input = np.random.randint(0, 2, size=(128, 10))
np.save('x_input.npy', x_input)
np.save('y_input.npy', y_input)
