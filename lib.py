import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def stylePlot(ax, legend: bool = True, legloc: str = "upper right", grid: bool = True):
    # ax.set_facecolor((0,0,0,0.04))
    # ax.set_aspect("equal")
    if grid:
        ax.grid(linestyle="dotted", color='gray', linewidth=0.4)
        ax.grid(which = "minor", linestyle="dotted", color='silver', linewidth=0.4)
        ax.minorticks_on()
    if legend:
        leg = ax.legend(ncol=1, frameon=True, loc=legloc, prop={'size': 10}, fancybox=True, framealpha=0.1,) #labelcolor='linecolor'
        leg.get_frame().set_boxstyle('Round', pad=0, rounding_size=0)
        # leg.get_frame().set_edgecolor("black")
        leg.get_frame().set_alpha(None)
    plt.tight_layout()

def euler(function, tspan:list=[0, 0], ic:list=[0, 0], h:float=0.1):
    X = np.arange(tspan[0], tspan[1], h)
    Y = np.zeros(len(X))
    Y[0] = ic[1]
    for i in range(0, len(Y)-1, 1):
        k1 = function(X[i], Y[i])
        Y[i + 1] = Y[i] + h*k1
    return X, np.array(Y)

def collatz(function, tspan:list=[0, 0], ic:list=[0, 0], h:float=0.1):
    X = np.arange(tspan[0], tspan[1], h)
    Y = np.zeros(len(X))
    Y[0] = ic[1]
    for i in range(0, len(Y)-1, 1):
        k1 = function(X[i], Y[i])
        k2 = function(X[i] + h/2, Y[i] + h/2*k1)
        Y[i + 1] = Y[i] + h*k2
    return X, np.array(Y)

def rk4(function, tspan:list=[0, 0], ic:list=[0, 0], h:float=0.1):
    X = np.arange(tspan[0], tspan[1], h)
    Y = np.zeros(len(X))
    Y[0] = ic[1]
    for i in range(0, len(Y)-1, 1):
        k1 = function(X[i], Y[i])
        k2 = function(X[i] + h/2, Y[i] + h/2*k1)
        k3 = function(X[i] + h/2, Y[i] + h/2*k2)
        k4 = function(X[i] + h, Y[i] + h*k3)
        Y[i + 1] = Y[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return X, np.array(Y)



def recog_type(eigVals):
    l1re = np.real(eigVals[0])
    l1im = np.imag(eigVals[0])
    l2re = np.real(eigVals[1])
    l2im = np.imag(eigVals[1])
    if l1im == 0. and l2im == 0.:
        if (l1re < 0 and l2re > 0) or (l1re > 0 and l2re < 0):
            return "sedlo"
        if (l1re < 0 and l2re < 0):
            return "stabilní uzel"
        if (l1re > 0 and l2re > 0):
            return "nestabilní uzel"
    else:
        if (abs(l1re) <= 1e-6 and abs(l2re) <= 1e-6):
            return "střed"
        elif (l1re < 0 and l2re < 0):
            return "stabilní ohnisko"
        elif (l1re > 0 and l2re > 0):
            return "nestabilní ohnisko"
    return "-"

def round_zero(number):
    if abs(float(number)) < 0.0001:
        return 0
    else: return number

def plot_eigen_vecs(eigVals, eigVecs, portrait_type, ax):
    if portrait_type in ["sedlo", "stabilní uzel", "nestabilní uzel", '-']:
        cols = ['aqua', 'springgreen']
        for i, eigVec in enumerate(eigVecs):
            ax.axline((0, 0), slope=eigVec[1]/eigVec[0], color=cols[i], lw=1.6,
                      label=f"$\\lambda_{i+1}={sp.latex(round_zero(sp.nsimplify(np.real(eigVals[i]))))}+{sp.latex(round_zero(sp.nsimplify(np.imag(eigVals[i]))))}j$".replace('+-', '-'), zorder=3
            )
            if np.real(eigVals[i]):
                fac = 1
                if np.real(eigVals[i]) < 0:
                    fac = -1
                for j in [1, -1]:
                    ax.arrow(j*5*eigVec[0], j*5*eigVec[1], j*1*fac*eigVec[0], j*1*fac*eigVec[1], shape='full', lw=0, length_includes_head=True, head_width=0.56, overhang=0.2, color=cols[i], zorder=3)

def plot_eigen_vals_positions(eigVals, ax):
    cols = ['aqua', 'springgreen']
    for i, eigVal in enumerate(eigVals):
        ax.scatter([sp.nsimplify(np.real(eigVal))], [sp.nsimplify(np.imag(eigVal))], color=(0,0,0,0), edgecolors=cols[i],
                   s=50, label=f"$\\lambda_{i+1}={sp.latex(round_zero(sp.nsimplify(np.real(eigVal))))}+{sp.latex(round_zero(sp.nsimplify(np.imag(eigVal))))}i$".replace('+-', '-')
        )

def tangent_vec(A, X0, ax):
    T = A@X0
    ax.plot([X0[0], X0[0]+T[0]], [X0[1], X0[1]+T[1]], color="white", lw=1.6, label="$\\tau$", zorder=3)
    ax.arrow(X0[0], X0[1], T[0], T[1], shape='full', lw=0, length_includes_head=True, head_width=0.56, overhang=0.2, color="white", zorder=3)
    ax.scatter([X0[0]], [X0[1]], s=42, color="white", label="$\\mathbf{X}_0$", marker='o', lw=2)

def collatz2t2(A, X0, step, steps):
    Y = np.zeros((steps, 2))
    Y[0] = X0
    for i in range(steps - 1):
        k1 = A@Y[i]
        Y[i+1] = Y[i] + step*k1
        if abs(Y[i+1][0]) >= 10 or abs(Y[i+1][1] > 10): Y[i+1] = Y[i]
    return Y

def unpack_eig_vals(eigVals):
    arr = []
    for val in eigVals:
        arr.append(abs(np.real(val)))
        arr.append(abs(np.imag(val)))
    return arr