# -*- coding: utf-8 -*-
"""
Created on Sun Jun 09 21:49:07 2019

@author: ChinnaRaj C
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

# Implement MLE to calibrate parameters
def VasicekCalibration_MLE(rates, dt=1.0/252):
    
    n = len(rates)
    
    Sx = sum(rates[0:(n-1)])
    Sy = sum(rates[1:n])
    Sxx = np.dot(rates[0:(n-1)], rates[0:(n-1)])
    Sxy = np.dot(rates[0:(n-1)], rates[1:n])
    Syy = np.dot(rates[1:n], rates[1:n])
    theta = (Sy * Sxx - Sx * Sxy) / (n * (Sxx - Sxy) - (Sx**2 - Sx*Sy))
    kappa = -np.log((Sxy - theta * Sx - theta * Sy + n * theta**2) / (Sxx - 2*theta*Sx + n*theta**2)) / dt
    a = np.exp(-kappa * dt)
    sigmah2 = (Syy - 2*a*Sxy + a**2 * Sxx - 2*theta*(1-a)*(Sy - a*Sx) + n*theta**2 * (1-a)**2) / n
    sigma = np.sqrt(sigmah2*2*kappa / (1-a**2))
    r0 = rates[n-1]
    
    return [kappa, theta, sigma, r0]




#### using least square method
    
def VasicekCalibration_OLS_new(rates, dt=1.0/252):
    rates=spot_rate['spot']
    n = len(rates)
    Sx = sum(rates[0:(n-1)])
    Sy = sum(rates[1:n])
    Sxx = np.dot(rates[0:(n-1)], rates[0:(n-1)])
        
    Sxy = np.dot(rates[0:(n-1)], rates[1:n])
    Syy = np.dot(rates[1:n], rates[1:n])
    
    dt=1.0/252
        
    a=(n*Sxy-Sx*Sy)/(n*Sxx-Sx**2)
    
    b=(Sy-a*Sx)/n
    
    sde= np.sqrt((n*Syy-Sy**2-a*(n*Sxy-Sx*Sy))/(n*(n-2)))
        
    kappa=-np.log(a)/dt
    
    theta=b/(1-a)
    
    sigma=sde*np.sqrt(-2*np.log(a)/(dt*(1-a**2)))
    
    r0 = rates.iloc[-1]

    return [kappa, theta, sigma, r0]



def VasicekCalibration_OLS(rates, dt=1.0/252):
    rates=spot_rate['spot']
    Sx = rates[0:-1]
    Sy = rates[1:]
    func=np.polyfit(Sx,Sy,1,full=True)
    
    dt=1.0/252    
    kappa=-np.log(func[0][0])/dt    
    theta=func[0][1]/(1-func[0][0])
    
    sigma=func[1][0]*np.sqrt(-2*np.log(func[0][0])/(dt*(1-func[0][0]**2)))    
    r0 = rates.iloc[-1]
    return [kappa, theta, sigma, r0]



def VasicekNextRate(r, kappa, theta, sigma, dt=1/252.0):
    # Implements above closed form solution
    
    val1 = np.exp(-1*kappa*dt)
    val2 = (sigma**2)*(1-val1**2) / (2*kappa)
    out = r*val1 + theta*(1-val1) + (np.sqrt(val2))*np.random.normal()
        
    out=r+kappa*(theta-r)*dt+sigma*np.sqrt(dt)*np.random.normal()
        
    return out

def VasicekSim(N, r0, kappa, theta, sigma, dt = 1/252.0):
    short_r = [0.0]*N
    short_r[0] = r0
    for i in range(1,N):
        short_r[i] = VasicekNextRate(short_r[i-1], kappa, theta, sigma, dt)
    return short_r

def VasicekMultiSim(M, N, r0, kappa, theta, sigma, dt = 1/252.0):
    sim_arr = np.ndarray((N, M))
    
    for i in range(0,M):
        sim_arr[:, i] = VasicekSim(N, r0, kappa, theta, sigma, dt)
    
    return sim_arr


def VasicekNegativeProb(Kappa, theta,sigma,r0,dt=1/252.0):
    
    variance=sigma**2*(1-np.exp(-2*kappa*dt))/(2*kappa)

    mean=r0*np.exp(-kappa*dt)+theta*(1-np.exp(-kappa*dt))
    
    z=(0-mean)/((variance*dt)**0.5)
    
    return norm.cdf(z)

""" Get zero coupon bond price by Vasicek model """
def exact_zcb(theta, kappa, sigma, tau, r0):
    B = (1 - np.exp(-kappa*tau)) / kappa
    
    beta=sigma**2
    nu=kappa*theta
    
    A= 1/kappa**2*(B-tau)*(nu*kappa-beta/2)-beta*B**2/(4*kappa)
    
    return np.exp(A-r0*B)


def zerocurve(zcbs,T):
    return -np.log(zcbs)/T


if __name__ == "__main__":
        
    path=r"User location where the spot rate file are placed"
    dat=pd.read_csv(path+"\Spot_Rates_V3.3.csv",skiprows=2)

    spot_rate=dat[dat.columns[1]][1:999]
    
    spot_rate=spot_rate.to_frame(name='spot')
    
    spot_rate=spot_rate.fillna(method='bfill')/100
    
    params_MLE = VasicekCalibration_MLE(spot_rate['spot'])
    
    params_OLS_new = VasicekCalibration_OLS_new(spot_rate['spot'])
    
    kappa = params_MLE[0]  #mean reversion strength
    theta = params_MLE[1]  #Long run mean
    sigma = params_MLE[2]  #volatility
    r0 = params_MLE[3]

    prob=VasicekNegativeProb(kappa, theta,sigma,r0,dt=1/252.0)
    
    years = 1
    N = years * 252
    t = np.arange(0,N)/252.0
    
    #A trajectory of the short rate
    
    test_sim = VasicekSim(N,r0,kappa, theta, sigma, 1.0/252)
    plt.plot(t,test_sim)
    plt.show()
   
    M = 100
    rates_arr = VasicekMultiSim(M, N,r0,kappa, theta, sigma)
    plt.plot(t,rates_arr)
    plt.hlines(y=theta, xmin = -100, xmax=100, zorder=10, linestyles = 'dashed', label='Theta',color='b')
    plt.annotate('Theta', xy=(1.0, theta+0.0005),color='b')
    plt.xlim(-0.05, 1.05)
    plt.ylabel("Rate")
    plt.xlabel("Time (yr)")
    plt.title("Simulated path of Short Rate using Vasicek Model")
    plt.show()
    
    Ts = np.r_[0.0:50.5:0.5]
    zcbs = [exact_zcb(theta,kappa,sigma,t,r0) for t in Ts]
    
    yield1=[zerocurve(zcbs[i],Ts[i]) for i in range(1,len(Ts))]

    plt.title("Zero Coupon Bond (ZCB) Values by Time")
    plt.plot(Ts, zcbs, label='ZCB')
    plt.ylabel("Value ($)")
    plt.xlabel("Time in years")
    plt.legend()
    plt.grid(True)
    plt.show()
    yield0=pd.DataFrame(list(zip(yield1)))
    
    plt.title("Zero Curve Rate by Time")
    plt.plot(Ts[1:], yield1, label='Yield')
    plt.ylabel("Yield (values)")
    plt.xlabel("Time in years")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    

