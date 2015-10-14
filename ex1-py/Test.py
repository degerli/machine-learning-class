import numpy as np
import matplotlib.pyplot as plt

x = np.arange(9).reshape(3, 3)
print(x)

print(np.where(x > 4))
