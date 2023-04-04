# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 13:46:21 2022

1D stress update program using return-mapping algorithm

Ramberg-osgood is used as the yield surface; associate flow rule; isotropic hardening law using power law

Kojic, M., & Bathe, K. J. (2005). Inelastic analysis of solids and structures (Vol. 2, No. 1, pp. 43). Berlin: Springer.

@author: Jiachen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class Strain_history:  #define strain path
    def __init__(self,step_size,no_step):
        self.step_size = step_size
        self.no_step = no_step
    def series(self):
        t = self.step_size * self.no_step
        t_hist = np.arange(0, t, self.step_size)
        e_hist=[]
        
        ##############triangle loading################
        #e1=np.arange(0, 0.02,0.001)
        #e2=np.arange(0.02, -0.02,-0.001)
        #e_2=np.append(e1,e2)
        #t_hist = np.linspace(0, 1,len(e_2))
        ##############################################
        
        ################sine loading###################
        #e_2 = - 0.008 * t_hist * np.sin(np.pi/2 * (t_hist))
        e_2 = (0.008 * t_hist) * signal.sawtooth(np.pi/2*(t_hist),0.5)
        ###############################################
        
        e_hist = np.append(e_hist, e_2)
        return t_hist, e_hist
    
    
class Ramberg_Osgood: #ramberg-osgood model F=|\sigma|-\sigma_y-(Cy*|ep|)^n
    def __init__(self, youngs, nu, sig_y, n, Cy):
        self.youngs = youngs
        self.nu = nu
        self.sig_y = sig_y
        self.n = n
        self.Cy = Cy
        
    def dFdsigma(self, sigma): #P
        if sigma > 0:
            output = 1
        else:
            output = -1
        return output
    def dFde(self, ep_abs): #ksi
        output = -self.Cy * self.n * (self.Cy * ep_abs)**(self.n - 1)
        #output = -100
        return output
 
    def beta(self): #
        output = 1
        return output    
 
    def F(self, sigma, ep_abs): 
        F = np.abs(sigma) - self.sig_y -  (self.Cy * ep_abs) ** self.n
        return F
    
    def update(self, sigma, e, ep, ep_abs, de):
        #first step, update total strain
        e = e + de
        #second step, elastic predictor
        ee = e - ep
        ep = ep
        sigma = sigma + self.youngs * de
        dlambda = 0
        dfds=0
        dfde=0
        flag=0
        f=0
        #third step, check for yield function
        while self.F(sigma, ep_abs) >=0.00001:
            dlambda = self.F(sigma, ep_abs)/(self.dFdsigma(sigma) * self.youngs * self.dFdsigma(sigma) - self.dFde(ep_abs) * self.beta())  
            dfds=self.dFdsigma(sigma)
            dfde=self.dFde(ep_abs)
            f=self.F(sigma, ep_abs)
            #ep = ep + dlambda * self.dFdsigma(sigma) oh my god!!
            ep = ep + dlambda * self.dFdsigma(sigma)
            ep_abs = ep_abs +dlambda *self.beta()
            sigma = sigma - dlambda * self.youngs * self.dFdsigma(sigma)  #something wrong with dlambda or ep
            flag=1
        return sigma, e, ep, ep_abs, dlambda, dfds, dfde, flag,f
                
class Isotropic:  #F=|\sigma|-\sigma_y-(100*|ep|)
    def __init__(self, youngs, nu, sig_y):
        self.youngs = youngs
        self.nu = nu
        self.sig_y = sig_y

    def F(self, sigma, ep_abs):
        F = np.abs(sigma)- self.sig_y - (100 * ep_abs)
        return F
        
    def dFdsigma(self, sigma):
        if sigma > 0:
            output = 1
        else:
            output = -1
        return output
    def dFde(self, ep_abs): #beta
        output = -100
        return output

    
    def update(self, sigma, e, ep, ep_abs, de):
        '''
        sigma: stress of the previous step
        e: total deformation in the previous step
        ep: total plastic deformation in the previous step
        de: strain increment
        epabs: absolute elastic strain
        '''
        #first step, update total strain
        e = e + de
        #second step, elastic predictor
        sigma = sigma + self.youngs * de
        dlambda = 0
        dfds=0
        dfde=0
        flag=0
        f=0
        #third step, check for yield function
        while self.F(sigma, ep_abs) >=0.00001:
            dlambda = self.F(sigma, ep_abs)/(self.dFdsigma(sigma) * self.youngs * self.dFdsigma(sigma) - self.dFde(ep_abs) * 1)  
            dfds=self.dFdsigma(sigma)
            dfde=self.dFde(ep_abs)
            f=self.F(sigma, ep_abs)
            ep_abs = ep_abs + dlambda * 1
            ep = ep + dlambda * self.dFdsigma(sigma)
            sigma = sigma - dlambda * self.youngs * self.dFdsigma(sigma)  #something wrong with dlambda or ep
            flag=1
        return sigma, e, ep, ep_abs, dlambda, dfds, dfde, flag,f


    
class Kinematic: 
    def __init__(self, youngs, nu, sig_y, c):
        self.youngs = youngs
        self.nu = nu
        self.sig_y = sig_y
        self.c = c

    def F(self, sigma, sig_back):
        F = np.abs(sigma - sig_back)- self.sig_y 
        return F
        
    def dFdsigma(self, sigma, sig_back):
        if sigma - sig_back > 0:
            output = 1
        else:
            output = -1
        return output
    def dFdsig_k(self, sigma, sig_back):
        if sigma - sig_back > 0:
            output = -1
        else:
            output = 1
        return output

    def beta(self, sigma, sig_back):
        if sigma - sig_back > 0:
            output = self.c
        else:
            output = -self.c
        return output
    
    def update(self, sigma, sig_back, e, ep, de):
        #first step, update total strain
        e = e + de
        #second step, elastic predictor
        sigma = sigma + self.youngs * de
        #third step, check for yield function
        while self.F(sigma, sig_back) >=0.00001:
            dlambda = self.F(sigma, sig_back)/(self.dFdsigma(sigma, sig_back) * self.youngs * self.dFdsigma(sigma, sig_back) - self.beta(sigma, sig_back) * self.dFdsig_k(sigma, sig_back))  
            sig_back = sig_back + dlambda * self.beta(sigma, sig_back)
            ep = ep + dlambda * self.beta(sigma, sig_back)/self.c
            sigma = sigma - dlambda * self.youngs * self.dFdsigma(sigma, sig_back)  #something wrong with dlambda or ep

        return sigma, sig_back, e, ep
    
        
path1 = Strain_history(0.002, 10000) #strain_path

t_hist, e_hist = path1.series() 

plt.figure()
plt.plot(t_hist, e_hist) #plot strain_path
plt.show()
iso = Isotropic(710, 0.33, 7.1)

stress = [0]
strain = [0]
ep = [0]
ep_abs =[0] #what's the relationship between ep and ep_abs
dlambda =[0]
dfdsigma =[0]
dfde =[0]
flag =[0]
f=[0]
for e in e_hist[1:]:
    strain.append(e)
    de = strain[-1] - strain[-2]
    sigma_1, e, ep_1, ep_abs_1, dlambda_1, dfdsigma_1, dfde_1, flag_1, f_1   = iso.update(stress[-1], strain[-1], ep[-1], ep_abs[-1], de)
    ep = np.append(ep, ep_1)
    ep_abs = np.append(ep_abs, ep_abs_1)
    dlambda = np.append(dlambda, dlambda_1)
    dfdsigma = np.append(dfdsigma, dfdsigma_1)
    dfde = np.append(dfde, dfde_1)
    stress = np.append(stress, sigma_1)
    flag = np.append(flag, flag_1)
    f=np.append(f, f_1)
plt.figure()
plt.plot(e_hist, stress) #plot strain_path
plt.show()



# kin = Kinematic(710, 0.33, 7.1, 100)

# stress = [0]
# stress_back=[0]
# strain = [0]
# ep = [0]
# for e in e_hist[1:]:
#     strain.append(e)
#     de = strain[-1] - strain[-2]
#     sigma_1, sig_back_1, e_1, ep_1   = kin.update(stress[-1], stress_back[-1], strain[-1], ep[-1], de)
#     ep = np.append(ep, ep_1)
#     stress_back = np.append(stress_back, sig_back_1)
#     stress = np.append(stress, sigma_1)
# plt.figure()
# plt.plot(e_hist, stress) #plot strain_path
# plt.show()



# ro = Ramberg_Osgood(710, 0.33, 7.1, 3, 10)

# stress = [0]
# strain = [0]
# ep = [0]
# ep_abs =[0]
# dlambda =[0]
# dfdsigma =[0]
# dfde =[0]
# flag =[0]
# f=[0]
# for e in e_hist[1:]:
#     strain.append(e)
#     de = strain[-1] - strain[-2]
#     sigma_1, e, ep_1, ep_abs_1, dlambda_1, dfdsigma_1, dfde_1, flag_1, f_1   = ro.update(stress[-1], strain[-1], ep[-1], ep_abs[-1],  de)
#     ep = np.append(ep, ep_1)
#     ep_abs = np.append(ep_abs, ep_abs_1)
#     dlambda = np.append(dlambda, dlambda_1)
#     dfdsigma = np.append(dfdsigma, dfdsigma_1)
#     dfde = np.append(dfde, dfde_1)
#     stress = np.append(stress, sigma_1)
#     flag = np.append(flag, flag_1)
#     f=np.append(f, f_1)
# plt.figure()
# plt.plot(e_hist, stress) #plot strain_path
# plt.show()
