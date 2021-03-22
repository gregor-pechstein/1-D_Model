
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt


def plot_detaW(path):
    Bx_Bt=np.arange(1,3,0.01)
    L=1
    zx=np.array([0.2,0.4,0.6])

    delta_n=np.zeros([Bx_Bt.shape[0],zx.shape[0]])

    for ii in np.arange(zx.shape[0]):
        delta_n[:,ii]=DetachmentWindow(Bx_Bt,zx[ii],L,1)

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(Bx_Bt,delta_n[:,0], label=r'$z_{x}/L=0.2$')
    plt.plot(Bx_Bt,delta_n[:,1], label=r'$z_{x}/L=0.4$')
    plt.plot(Bx_Bt,delta_n[:,2], label=r'$z_{x}/L=0.6$')
    plt.xlim(1,3)
    plt.ylim(0)
    plt.grid(alpha=0.5)
    plt.xlabel(r'Bx/Bt', fontsize=18)
    plt.ylabel(r'$\Delta n_{u}$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/delta_n.png', dpi=300)
    plt.show()

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(Bx_Bt,delta_n[:,0]/delta_n[0,0], label=r'$z_{x}/L=0.2$')
    plt.plot(Bx_Bt,delta_n[:,1]/delta_n[0,1], label=r'$z_{x}/L=0.4$')
    plt.plot(Bx_Bt,delta_n[:,2]/delta_n[0,2], label=r'$z_{x}/L=0.6$')
    plt.xlim(1,3)
    plt.ylim(0)
    plt.grid(alpha=0.5)
    plt.xlabel(r'Bx/Bt', fontsize=18)
    plt.ylabel(r'$\Delta n_{u}/\Delta n_{u}(B_{x}/B_{t}=1)$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/delta_n_relativ.png', dpi=300)
    plt.show()
    
    return

def plot_n_fI_S0(path):

    Bx_Bt=np.arange(1,3,0.01)
    L=1
    zx=0.2
    beta=np.array([1,2,7/5])

    delta_C=np.zeros([Bx_Bt.shape[0],beta.shape[0]])

    for ii in np.arange(beta.shape[0]):
        delta_C[:,ii]=DetachmentWindow(Bx_Bt,zx,L,beta[ii])

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(Bx_Bt,delta_C[:,0], label=r'$n_{u}$')
    plt.plot(Bx_Bt,delta_C[:,1], label=r'$f_{I}$')
    plt.plot(Bx_Bt,delta_C[:,2], label=r'$S_{0}$')
    plt.xlim(1,3)
    #plt.ylim(0)
    plt.grid(alpha=0.5)
    plt.xlabel(r'Bx/Bt', fontsize=18)
    plt.ylabel(r'$\Delta C$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/delta_Cpaper.png', dpi=300)
    plt.show()

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(Bx_Bt,delta_C[:,0]/delta_C[0,0], label=r'$n_{u}$')
    plt.plot(Bx_Bt,delta_C[:,1]/delta_C[0,1], label=r'$f_[I}$')
    plt.plot(Bx_Bt,delta_C[:,2]/delta_C[0,2], label=r'$S_{0}$')
    plt.xlim(1,3)
    plt.ylim(0)
    plt.grid(alpha=0.5)
    plt.xlabel(r'Bx/Bt', fontsize=18)
    plt.ylabel(r'$\Delta C$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/delta_C_normilized.png', dpi=300)
    plt.show()

    return Bx_Bt, delta_C

def stability(path):
    Bx_Bt=np.arange(0,1,0.01)
    L=1

    zx=3/2 * L*((abs(1/Bx_Bt))**(7/2)-1) * (1 + abs(1/Bx_Bt) +(abs(1/Bx_Bt))**2 + 3/2 * ((abs(1/Bx_Bt))**(7/2) -1 ))**(-1)

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(zx,Bx_Bt, label=r'stability limit')
    plt.grid(alpha=0.5)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel(r'Bx/Bt', fontsize=18)
    plt.xlabel(r'$z_{x}/L$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper right', framealpha=0, fontsize=12)
    plt.savefig(path+'/stability.png', dpi=300)
    plt.show()
    return

    
def DetachmentWindow(Bx_Bt, zx,L, beta):
    
    delta_n =( Bx_Bt * (2*zx/(3* (L-zx)) *(1 + abs(1/Bx_Bt) + (abs(1/Bx_Bt))**2 ) +1 )**(2/7))**beta -1

    return delta_n

def qi_qf(Bx_Bt,zxRelativ,coulomb,path):
    
    
    Tc=5
    Th=65
    nu_x=1e20
    fI=0.04
    L=26.5
    zh=np.arange(0,L,0.01)
    zx=zxRelativ*L
    me = 9.1093837015*1e-31
   # coulomb =10

    tau, kappa_par = thermalconductivity(nu_x,me, coulomb)
    U =U_func( Tc,Th, kappa_par)
    S0=S0x_func(L,zx, U,fI,nu_x)
    
    qi= qi_func(zh,S0,L,zx)

    B_Bx=B_Bx_funx(zh, zx,Bx_Bt)
    Tu =Tu_func(S0,L,zx,kappa_par,zh, B_Bx)
    qf_x= qf_func(kappa_par,fI,nu_x,U, zh,B_Bx,S0,L,zx)
    beta= 1
    delta_n=DetachmentWindow(Bx_Bt, zx,L, beta)
    nu_t=nu_x/(delta_n+1)
    qf_t=qf_func(kappa_par,fI,nu_t,U, zh,B_Bx,S0,L,zx)

    plot_qi_qf(path,zh,L,qf_x,qf_t,qi,Tu,Bx_Bt)
    
    return qi, qf_x,qf_t, Tu, S0, L,zh, zx, U, B_Bx, kappa_par,nu_x,nu_t,delta_n, fI

def plot_qi_qf(path,zh,L,qf_x,qf_t,qi,Tu,Bx_Bt):
    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(zh/L,qf_x, label=r'$qf_{x}$')
    plt.plot(zh/L,qf_t, label=r'$qf_{t}$')
    plt.plot(zh/L,qi, label=r'qi')
    plt.grid(alpha=0.5)
    plt.xlim(0,1)
    plt.ylim(plt.ylim()[0], 1.0)
    plt.ylabel(r'Power flux', fontsize=18)
    plt.xlabel(r'$z_{h}/L$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/qf_qi'+str(Bx_Bt)+'.png', dpi=300)
    plt.show()

    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
    plt.plot(zh/L,Tu, label=r'$T_{u}$')

    plt.grid(alpha=0.5)
    plt.xlim(0,1)
    plt.ylim(0)
    plt.ylabel(r'T in [eV]', fontsize=18)
    plt.xlabel(r'$z_{h}/L$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='upper right', framealpha=0, fontsize=12)
    plt.savefig(path+'/Tu'+str(Bx_Bt)+'.png', dpi=300)
    plt.show()
    
    return

def S0x_func(L,zx,U,fI,nu):
    S0x=1/(L-zx)*(U*np.sqrt(fI)*nu*((L-zx)/2)**(2/7))**(7/5)
    return S0x

def qi_func(zh, S0,L,zx):
    qi=np.zeros(zh.shape[0])
    
    for ii in range(zh.shape[0]):
        if (zh[ii]<zx):  
            qi[ii]=- S0*(L - zx)
        else:
            qi[ii]= -S0*(L - zh[ii])

    return qi


def qf_func(kappa_par,fI,nu,U, zh, B_Bx,S0,L,zx):

    qf=np.zeros(zh.shape[0])
    for ii in range(zh.shape[0]):
        if (zh[ii]<zx):
            qf[ii] = -U *np.sqrt(fI) *nu * 1/B_Bx[ii] *(S0 *(L-zx))**(2/7) *  ((zx-zh[ii])/3 *(1 + abs(B_Bx[ii]) +(abs(B_Bx[ii]))**2 ) + (L-zx)/2)**(2/7)
        else:
            qf[ii] = -U * np.sqrt(fI) * nu * 1/B_Bx[ii] * (S0/2)**(2/7) *(L-zh[ii])**(4/7)
    return qf

def Q(T=np.arange(0,120,0.1)):

    Q= 5.9*1e-34 *(T-1)**0.5 *(80-T)/(1+3.1*1e-3*(T-1)**2)*np.heaviside(80-T,1)*np.heaviside(-1+T,1)
    


    return Q


def B_Bx_funx(zh, zx,Bx_Bt):

    B_Bx=np.zeros(zh.shape[0])
    
    for ii in range(zh.shape[0]):
        if (zh[ii]<zx):  
            B_Bx[ii]= 1/Bx_Bt + (1-1/Bx_Bt) *zh[ii]/zx
        else:
            B_Bx[ii]= 1

    return B_Bx

            

def U_func( Tc,Th, kappa_par):
    u = lambda T: T**0.5 * 5.9*1e-34 *(T-1)**0.5 *(80-T)/(1+3.1*1e-3*(T-1)**2)*np.heaviside(80-T,1)*np.heaviside(-1+T,1)
    RadiationInt= quad(u, Tc,Th)
    kB= 8.617333262145*1e-5
    
    U=7**(2/7) *(2 *kappa_par)**(3/14) * np.sqrt(RadiationInt[0])

    return U



def Tu_func(S0,L,zx,kappa_par,zh, B_Bx):

    Tu=np.zeros(zh.shape[0])
    for ii in range(zh.shape[0]):
        if (zh[ii]<zx):  
            Tu[ii]=(7*S0*( L-zx)/(2*kappa_par))**(2/7) * ((zx-zh[ii])/3 *(1 + abs(B_Bx[ii]) +(abs(B_Bx[ii]))**2 ) + (L-zx)/2)**(2/7)
        else:
            Tu[ii]= (7*S0/(4*kappa_par))**(2/7) *(L-zh[ii])**(4/7)

    return Tu



#def S0_func(Tu,L,zx,k1,zh):
#
#    S0=np.zeros([Tu.shape[0],zh.shape[0]])
#    for ii in range(zh.shape[0]):
#        if (zh[ii]<zx):  
#            S0[:,ii]=Tu**(7/2)*2*k1/(7*(L-zx)*(zx-zh[ii] +(L-zx)/2))
#        else:
#            S0[:,ii]= Tu**(7/2)*k1*4/(7*(L-zh[ii])**2)
#
#    return S0
    

def thermalconductivity(n,me, coulomb):
    kB= 8.617333262145*1e-5
    tau=3.5*1e5/(coulomb*n)
    kappa_par= 3.16 * n*tau/me
    #omega=qe*B/me
    #kappa_perp = 4.66*n/(me*omega**2*tau)
    return tau, kappa_par
