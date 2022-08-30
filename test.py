import numpy as np
import matplotlib.pyplot as plt


x = [0, 2, 2, 8, 8, 24, 24, 24, 72, 72, 100, 100, 144, 144]

y = [4.562051e-04, 2.704509e-03, 8.384947e-04, 2.048477e-03,
     1.711415e-03, 3.355744e-03, 1.325871e-03, 3.479523e-03,
     4.632386e-03, 4.374194e-03, 6.270893e-03, 5.153824e-03,
     5.921773e-03, 4.413801e-03]

coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef)

plt.plot(x, y, 'bo', x, poly1d_fn(x), '--k')
plt.title('Calibration curve')
plt.xlabel('Concentration [pM]')
plt.ylabel('Slope shift [nm]')
plt.show()