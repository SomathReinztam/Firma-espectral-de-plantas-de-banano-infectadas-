

import sys

pth = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\code3\3_analisis_datos"
pth.replace('\\', '/')

pth2 = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\code3\9_LinearRegression"
pth2 = pth2.replace('\\', '/')

sys.path.append(pth)
sys.path.append(pth2)
"""
from plot_PCA_2dim import plot_1_datosPCA_2d

plot_1_datosPCA_2d()

"""

from plot_PCA_2dim import plot_1_datosPCA_2d
from Plot_Ridge_LinearRegression.py import plot_dpi_test

plot_dpi_test()

