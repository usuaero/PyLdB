# -*- coding: utf-8 -*-
"""
PyLdB
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.integrate as integrate
import time
#plt.style.use('//home//christian//.cache//matplotlib//aiaapaper.mplstyle')

def Import(filename):
    #Import time and pressure data from file
#    if filename is not None:
#    Data = np.genfromtxt(filename, skip_header = 2)
#    Time = Data[:36002,0]*10**3
#    Pressure = Data[:36002,1]*0.021
    Data = np.genfromtxt(filename, skip_header = 3)
    Time = Data[:,0]
    Pressure = Data[:,1]
    #http://www.unece.org/fileadmin/DAM/trans/main/wp29/wp29wgs/wp29grsp/QRTV-09-05.pdf
    return Time,Pressure

def Tables():
    Leqrange = np.arange(1,141,1)
    SonesL = [0.078,0.087,0.097,0.107,0.118,
              0.129,0.141,0.153,0.166,0.181,
              0.196,0.212,0.230,0.248,0.269,
              0.290,0.314,0.339,0.367,0.396,
              0.428,0.463,0.500,0.540,0.583,
              0.630,0.680,0.735,0.794,0.857,
              0.926,1.000,1.080,1.170,1.260,
              1.360,1.470,1.590,1.710,1.850,
              2.000,2.160,2.330,2.520,2.720,
              2.940,3.180,3.430,3.700,4.000,
              4.320,4.670,5.040,5.440,5.880,
              6.350,6.860,7.410,8.000,8.640,
              9.330,10.10,10.90,11.80,12.70,
              13.70,14.80,16.00,17.30,18.70,
              20.20,21.80,23.50,25.40,27.40,
              29.60,32.00,34.60,37.30,40.30,
              43.50,47.00,50.80,54.90,59.30,
              64.00,69.10,74.70,80.60,87.10,
              94.10,102.0,110.0,119.0,128.0,
              138.0,149.0,161.0,174.0,188.0,
              203.0,219.0,237.0,256.0,276.0,
              299.0,323.0,348.0,376.0,406.0,
              439.0,474.0,512.0,553.0,597.0,
              645.0,697.0,752.0,813.0,878.0,
              948.0,1024.,1106.,1194.,1290.,
              1393.,1505.,1625.,1756.,1896.,
              2048.,2212.,2389.,2580.,2787.,
              3010.,3251.,3511.,3792.,4096.]
    SonesF = np.copy(SonesL[9:104])
    F = [0.100,0.122,0.140,0.158,0.174,
         0.187,0.200,0.212,0.222,0.232,
         0.241,0.250,0.259,0.267,0.274,
         0.281,0.287,0.293,0.298,0.303,
         0.308,0.312,0.316,0.319,0.320,
         0.322,0.322,0.320,0.319,0.317,
         0.314,0.311,0.308,0.304,0.300,
         0.296,0.292,0.288,0.284,0.279,
         0.275,0.270,0.266,0.262,0.258,
         0.253,0.248,0.244,0.240,0.235,
         0.230,0.226,0.222,0.217,0.212,
         0.208,0.204,0.200,0.197,0.195,
         0.194,0.193,0.192,0.191,0.190,
         0.190,0.190,0.190,0.190,0.190,
         0.191,0.191,0.192,0.193,0.194,
         0.195,0.197,0.199,0.201,0.203,
         0.205,0.208,0.210,0.212,0.215,
         0.217,0.219,0.221,0.223,0.224,
         0.225,0.226,0.227,0.227,0.227]
#    f_center = [1.0000001,1.25,1.6,2.0,2.5,3.15,
#                4.,5.,6.3,8.,10.,
#                12.5,16.,20.,25.,31.5,
#                40.,50.,63.,80.,100.,
#                125.,160.,200.,250.,315.,
#                400.,500.,630.,800.,1000.,
#                1250.,1600.,2000.,2500.,3150.,
#                4000.,5000.,6300.,8000.,10000.,
#                12500.]
#    f_l = [0.89,1.12,1.41,1.78,2.24,2.82,
#           3.55,4.47,5.62,7.08,8.91,
#           11.2,14.1,17.8,22.4,28.2,
#           35.5,44.7,56.2,70.8,89.1,
#           112.,141.,178.,224.,282.,
#           355.,447.,562.,708.,891.,
#           1120.,1410.,1780.,2240.,2820.,
#           3550.,4470.,5620.,7080.,8910.,
#           11200.]
#    f_u = [1.12,1.41,1.78,2.24,2.82,
#           3.55,4.47,5.62,7.08,8.91,
#           11.2,14.1,17.8,22.4,28.2,
#           35.5,44.7,56.2,70.8,89.1,
#           112.,141.,178.,224.,282.,
#           355.,447.,562.,708.,891.,
#           1120.,1410.,1780.,2240.,2820.,
#           3550.,4470.,5620.,7080.,8910.,
#           11200.,14100.]
    f_center = [1.000000000001,1.26,1.58,2.0,2.51,3.16,
                3.98,5.01,6.31,7.94,10.,
                12.59,15.85,19.95,25.12,31.62,
                39.81,50.12,63.10,79.43,100.,
                125.89,158.49,199.53,251.19,316.23,
                398.11,501.19,630.96,794.33,1000.,
                1258.9,1584.9,1995.3,2511.9,3162.3,
                3981.1,5011.9,6309.6,7943.3,10000.,
                12589.3]
    f_l = np.zeros(len(f_center))
    f_u = np.zeros(len(f_center))
    for i in range(42):
        f_l[i] = f_center[i]/(2**(1/6))
        f_u[i] = f_center[i]*(2**(1/6))
#    fci = 1.00001
#    f_center = np.zeros(42)
#    f_l = np.zeros(42)
#    f_u = np.zeros(42)
#    for i in range(42):
#        f_center[i] = fci*(2**(i/3))
#        f_l[i] = f_center[i]/(2**(1/6))
#        f_u[i] = f_center[i]*(2**(1/6))
    return Leqrange, SonesL, SonesF, F, f_center, f_l, f_u
"""

"""

def SigProc(T,P):
    Leqrange,SonesL,SonesF,F, f_center,f_l,f_u = Tables()
    plt.rcParams["text.usetex"] = True
    font = {'family': 'serif',    
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
#    plt.figure(0, figsize = (4.25*2.,2.125*2.))
#    plt.plot(T,P, color = "black")
#    plt.rc('axes', linewidth = 2)
#    plt.tick_params(direction = 'in')
#    plt.xlabel('Time, ms',fontdict=font)
#    plt.xlim((T.min(),T.max()))
#    plt.xticks(family = 'serif',fontsize = 12)
#    plt.ylabel(r'$\Delta{p}$, lb/{ft^2}',fontdict=font)
#    plt.ylim((P.min()+P.min()/10.,P.max()+P.max()/10.))
#    plt.yticks(family = 'serif',fontsize = 12)
#    plt.savefig("signature.pdf",dpi = 2000)
    P = window(P,800)
    pcheck = integrate.trapz(np.abs(P)**2,T*10**-3)
    fp = len(P)*10
    rp = len(P)*10
#    plt.figure(1)
#    plt.plot(T,P)
#    plt.xlabel('Time, ms')
#    plt.ylabel(r'Pressure, lb/{ft^2}')
#    plt.xlim((T.min(),T.max()))
#    plt.ylim((P.min()+P.min()/10.,P.max()+P.max()/10))
#    plt.savefig("test.pdf")
#    plt.legend(loc = 'upper right')
    T,P = padding(T,P,fp,rp)
    N = len(P)
    dt = ((T[-1]-T[0])/N)*(10**-3)
    po = (2.900755e-9)*144 
    FFT = np.fft.fft(P)
    freq = np.fft.fftfreq(N)/dt
    freqOneSide = np.copy(freq[0:N//2-1]) #Convert double sided Power Spectrum to single sided
    Power = (np.abs(FFT)**2)*(dt**2)
    Echeck = 2.*integrate.trapz(Power[:N//2-1],x = freq[:N//2-1])
#    plt.figure(2, figsize = (4.25*2.,2.125*2.))
#    plt.plot(freq,Power, color = "black")
#    plt.locator_params(axis = 'y',nbins = 5)
#    plt.rc('axes', linewidth = 2)
#    plt.tick_params(direction = 'in')
#    plt.xlabel('Frequency, Hz',fontdict=font)
#    plt.xticks(family = 'serif',fontsize = 12)
#    plt.xlim((-215.0,215.0))
#    plt.ylabel(r'$S_{xx}$',fontdict=font)
#    plt.ylim((0.0,Power.max()+Power.max()/10.))
#    plt.yticks(family = 'serif',fontsize = 12)
#    plt.savefig("ExPowerSpec.pdf",dpi = 2000)
    Power[1:N//2-1] = 2.*Power[1:N//2-1] #Convert double sided Power Spectrum to single sided 
    PowerOneSide = np.copy(Power[0:N//2-1])
    E = np.zeros(len(f_center))
    InterpFreq = np.append(f_l,f_u[-1])
    InterpPow = np.interp(InterpFreq,freqOneSide,PowerOneSide)
    OrigPow_Freq = np.array([freqOneSide,PowerOneSide])
    InterpPow_Freq = np.array([InterpFreq,InterpPow])
    FullPowFreq = np.concatenate((OrigPow_Freq,InterpPow_Freq),axis = 1)
    FullPowFreqSort = np.argsort(FullPowFreq[0])
    freqOne = FullPowFreq[0,FullPowFreqSort]
    PowerOne = FullPowFreq[1,FullPowFreqSort]
    for j in range(len(f_center)):
        sectionloc = np.nonzero((f_l[j] <= freqOne) & (freqOne <= f_u[j]))
        if len(sectionloc[0]) != 0:
            Pow_range = PowerOne[sectionloc[0][0]:sectionloc[0][-1]+1]
            f_range = (freqOne[sectionloc[0][0]:sectionloc[0][-1]+1])
            E[j] = integrate.trapz(Pow_range,x=f_range)
    print(pcheck,Echeck)
    print(100.*np.abs(Echeck-pcheck)/Echeck,"%")
    E /= 0.07
    L = 10*np.log10(E/(po**2)) - 3  #Definition of dB is based on some reference pressure
#    plt.figure(3, figsize = (4.25*2.,2.125*2.))
#    plt.scatter(f_center,L, color = "black")
#    plt.locator_params(axis = 'y',nbins = 5)
#    plt.tick_params(axis = 'x',which = 'minor', bottom = 'off')
#    plt.tick_params(axis = 'x',which = 'major', direction = 'in')
#    plt.tick_params(axis = 'y',which = 'major', direction = 'in')
#    plt.xscale('log')
#    plt.rc('axes', linewidth = 2)
#    plt.tick_params(direction = 'in')
#    plt.xlabel('Frequency, Hz',fontdict=font)
#    plt.xticks(family = 'serif',fontsize = 12)
#    plt.xlim((0.0,f_u[-1]))
#    plt.ylabel(r'$L_{p}$, dB',fontdict=font)
#    plt.ylim((L.min() + L.min()/10.,L.max()+L.max()/10.))
#    plt.yticks(family = 'serif',fontsize = 12)
#    plt.savefig('ExLp.pdf',dpi = 2000)
    print(E)
    Leq = np.zeros(len(L))
    for i in range(len(L)):
        if i > 39:
            Leq[i] = L[i] + 4.*(39. - i)
        if 35 <= i <= 39:
            Leq[i] = L[i]
        if 32 <= i <= 34:
            Leq[i] = L[i] - 2.*(35. - i)
        if 26 < i <= 31:
            Leq[i] = L[i] - 8.0
        if 20 <= i <= 26:
            if i == 26:
                Leq[i] = LoudLimits400(f_center[i],76.0,121.0,L[i],0.0)
            if i == 25:
                Leq[i] = LoudLimits400(f_center[i],77.5,122.5,L[i],1.5)
            if i == 24:
                Leq[i] = LoudLimits400(f_center[i],79.0,124.0,L[i],3.0)
            if i == 23:
                Leq[i] = LoudLimits400(f_center[i],80.5,125.5,L[i],4.5)
            if i == 22:
                Leq[i] = LoudLimits400(f_center[i],82.0,127.0,L[i],6.0)
            if i == 21:
                Leq[i] = LoudLimits400(f_center[i],83.5,128.5,L[i],7.5)
            if i == 20:
                Leq[i] = LoudLimits400(f_center[i],85.0,130.0,L[i],9.0)
        if i <= 19:
            LeqB = 160.0 - ((160.0 - L[i])*np.log10(80.0))/np.log10(f_center[i])
            X = 10.5
            f = 80.
            Leq[i] = LoudLimits400(f,86.5,131.5,LeqB,X)
    Sones = np.interp(Leq,Leqrange,SonesL,left = 0.0)
#    plt.figure(4, figsize = (4.25*2.,2.125*2.))
#    plt.scatter(f_center,Leq, color = "black")
#    plt.locator_params(axis = 'y',nbins = 5)
#    plt.tick_params(axis = 'x',which = 'minor', bottom = 'off')
#    plt.tick_params(axis = 'x',which = 'major', direction = 'in')
#    plt.tick_params(axis = 'y',which = 'major', direction = 'in')
#    plt.xscale('log')
#    plt.rc('axes', linewidth = 2)
#    plt.xlabel('Frequency, Hz',fontdict=font)
#    plt.xticks(family = 'serif',fontsize = 12)
#    plt.xlim((10**0, f_u[-1]))
#    plt.ylabel(r'$L_{eq}$, dB',fontdict=font)
#    plt.yticks(family = 'serif',fontsize = 12)
#    plt.ylim((Leq.min() + Leq.min()/10.,Leq.max()+Leq.max()))
#    plt.savefig('ExLeq.pdf',dpi = 2000)
#    plt.figure(5, figsize = (4.25*2.,2.125*2.))
#    plt.scatter(f_center,Sones, color = "black")
#    plt.locator_params(axis = 'y',nbins = 5)
#    plt.tick_params(axis = 'x',which = 'minor', bottom = 'off')
#    plt.tick_params(axis = 'x',which = 'major', direction = 'in')
#    plt.tick_params(axis = 'y',which = 'major', direction = 'in')
#    plt.xscale('log')
#    plt.rc('axes', linewidth = 2)
#    plt.xlabel('Frequency, Hz',fontdict=font)
#    plt.xticks(family = 'serif',fontsize = 12)
#    plt.xlim((0.0, f_u[-1]))
#    plt.ylabel(r'$S$, Sones',fontdict=font)
#    plt.yticks(family = 'serif',fontsize = 12)
#    plt.ylim((-0.35,Sones.max()+Sones.max()/10.))
#    plt.savefig('ExSones.pdf',dpi = 2000)
    Fmax = np.interp(Sones.max(),SonesF,F, left = 0.0,right = F[-1])
    S_t = Sones.max() + Fmax*(sum(Sones) - Sones.max())
    print(Leq)
    PLdB = 32.0 + 9.0*np.log2(S_t)
    print(PLdB, "PLdB")
    return PLdB

def window(P,xpoints):
    win = np.hanning(xpoints*2)
    P[:xpoints] *= win[:xpoints]
    P[-xpoints:] *= win[xpoints:]
    return P

def padding(T,P,fp,rp):
    P = np.pad(P, (fp, rp), 'constant')
    frontpad = fp*(T[1] - T[0])
    rearpad = T[-1] + rp*(T[1] - T[0]) + frontpad
    fronttime = np.linspace(T[0],frontpad,num = fp,endpoint = True)
    reartime = np.linspace(T[-1]+frontpad,rearpad, num = rp,endpoint = True)
    T[:] += frontpad
    T = np.insert(T,0,fronttime)
    T = np.append(T,reartime)
    return T,P

def LoudLimits400(fc,L_l,L_u,L,X):
    if L <= L_l:
        A = 115.0 - ((115.0-L)*np.log10(400.0))/np.log10(fc)
        Leq = A - 8.0
    if L_l < L <= L_u:
        Leq = L - X - 8.0
    if L > L_u:
        A = 160.0 - ((160.0 - L)*np.log10(400.0))/np.log10(fc)
        Leq = A - 8.0
    return Leq
    
def PyLdB(T,P):
    t0 = time.time() 
    np.set_printoptions(threshold=np.nan)    
    PLdB = SigProc(T,P)
    t1 = time.time()
    print(t1-t0)
    return PLdB

T,P = Import("panair_r1.sig")
PyLdB(T,P)
#Percevals(T,P)