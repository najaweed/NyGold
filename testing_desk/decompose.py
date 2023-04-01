import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pwlf

xdf = pd.read_csv('gold.csv', index_col='time', parse_dates=True)
xdf = xdf.iloc[5 * 40 - 2:8 * 40 - 2, :]
y = xdf.close.to_numpy()
x = np.linspace(0, len(y), len(y))
# initialize piecewise linear fit with your x and y data
my_pwlf = pwlf.PiecewiseLinFit(x, y, degree=1)

# fit the data for four line segments
res = my_pwlf.fitfast(5)
print(res)
# predict for the determined points
xHat = np.linspace(min(x), max(x), num=10000)
yHat = my_pwlf.predict(x)

# plot the results
plt.figure()
plt.plot(x, y, 'o')
plt.plot(x, yHat, '-')
plt.show()  # print(x)

# def time_series_embedding(data, delay=1, dimension=2):
#     "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
#     if delay * dimension > len(data):
#         raise NameError('Delay times dimension exceed length of data!')
#     emd_ts = np.array([data[0:len(data) - delay * dimension]])
#     for i in range(1, dimension + 1):
#         emd_ts = np.append(emd_ts, [data[i * delay:len(data) - delay * (dimension - i)]], axis=0)
#     return emd_ts
#
#
# num_emb = 60
# emb_x = time_series_embedding(x, 1, num_emb)
# u, s, v_h = np.linalg.svd(emb_x, full_matrices=False)
#
#
# def mode(r):
#     u_r, s_r, v_h_r = u[:, r - 1:r], s[r - 1:r], v_h[r - 1:r, :]
#     xx2 = np.matmul(np.matmul(u_r, np.diag(s_r)), v_h_r)
#     return xx2[-1, :]
#
#
# x1 = mode(r=1)
# x2 = mode(r=2)
# x3 = mode(r=3)
# x4 = mode(r=4)
# x5 = mode(r=5)
# x6 = mode(r=6)
# plt.figure(12)
# plt.plot(x[num_emb:])
# plt.plot(x1)
# plt.plot(x1 + x2)
# plt.plot(x1 + x2 + x3)
# plt.plot(x1 + x2 + x3 + x4)
# plt.plot(x1 + x2 + x3 + x4 + x5)
# plt.plot(x1 + x2 + x3 + x4 + x5 + x6)
# plt.show()
