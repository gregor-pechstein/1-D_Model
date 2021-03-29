
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

    
def DetachmentWindow(Bx_Bt, zx,L, beta,zh=0):
    
    delta_n =( Bx_Bt * (2*(zx-zh)/(3* (L-zx)) *(1 + abs(1/Bx_Bt) + (abs(1/Bx_Bt))**2 ) +1 )**(2/7))**beta -1

    return delta_n

def qi_qf(Bx_Bt,zxRelativ,coulomb,path):
    
    
    #Tc=5
    #Th=65
    nu_x=1e20
    fI=0.04
    L=26.5
    zh=np.arange(0,L,0.01)
    zx=zxRelativ*L
    me = 9.1093837015*1e-31
   # coulomb =10

    tau, kappa_par = thermalconductivity(nu_x,me, coulomb)
    U =U_func( 5,65, kappa_par)
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

def plot_qi_qf(path,zh,L,qf_t,qi,Tu,Bx_Bt):
    plt.rc('font', family='Serif')
    plt.figure(figsize=(8,4.5))
   # plt.plot(zh/L,qf_x, label=r'$qf_{x}$')
    plt.plot(zh/L,qf_t, label=r'$qf$')
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

    #plt.rc('font', family='Serif')
    #plt.figure(figsize=(8,4.5))
    #plt.plot(zh/L,Tu, label=r'$T_{u}$')

    #plt.grid(alpha=0.5)
    #plt.xlim(0,1)
    #plt.ylim(0)
    #plt.ylabel(r'T in [eV]', fontsize=18)
    #plt.xlabel(r'$z_{h}/L$', fontsize=18)
    #plt.tick_params('both', labelsize=14)
    #plt.tight_layout()
    #plt.legend(fancybox=True, loc='upper right', framealpha=0, fontsize=12)
    #plt.savefig(path+'/Tu'+str(Bx_Bt)+'.png', dpi=300)
    #plt.show()
    
    return


def q_func(Bx_Bt, Tu=110, zxRelativ=0.2, nu_x=1e20, fI=0.04,  L=26.5, zhRelativ=0.1):
    """treat S0x as a dependent parameter to a given nux and fIx"""
    me = 9.1093837015*1e-31
    coulomb =10

    z= np.arange(0,L,0.01)
    zh=zhRelativ*L
    zx=zxRelativ*L
    if zh> zx:
        print('unstable zh>zx')
        return

    B_Bx=B_Bx_funx(z, zx,Bx_Bt)
    B_Bx_zh =B_Bx[np.where(z==np.round(zh,2))]

    tau, kappa_par = thermalconductivity(nu_x,me, coulomb)
    U =U_func(5,65, kappa_par)
    S0x= S0x_func2(L,zx,fI,nu_x,Tu,kappa_par,U)
    #S0x=S0x_func(L,zx, U,fI,nu_x)

    #keep fI and SO constant vary only nu(zh)
    beta= 1
    delta_nh=DetachmentWindow(1/B_Bx_zh, zx,L, beta,zh)
    nu_h=nu_x/(delta_nh+1)

    qi= qi_zh(zh,S0x,L,zx)
    qf=qf_zh(kappa_par,fI,nu_h,U, zh,B_Bx_zh,S0x,L,zx)

    Tu_zh =Tuupstream(S0x,L,zx,kappa_par,zh,B_Bx_zh)
   # plot_qi_qf(path,z,L,qf,qi,Bx_Bt)

    T= np.zeros(z.shape[0])
    for ii in range(z.shape[0]):
        if (z[ii]>=zh):
            if (z[ii]>= zx):
                T[ii]=(Tu**(7/2)-7*S0x/(4*kappa_par) *(L-z[ii])**2)**(2/7)
            elif (z[ii]<zx):
                T[ii]= (Tu**(7/2) -(7*S0x*(L-zx)/(2*kappa_par))*((zx-z[ii])/3 * (1 + abs(B_Bx[ii]) +(abs(B_Bx[ii]))**2 ) + (L-zx)/2))**(2/7)
            else:
                print(error)
                return
            
        elif (z[ii]<zh):
            
            T[ii]= Tfront(z[ii])
        else:
            print(error2)

    H= np.zeros(z.shape[0])       
    for ii in range(z.shape[0]):
        if (z[ii]<zh):  
            H[ii]= nu_h**2 * Tu**2 * fI * Q(T[ii])/T[ii]**2
        else:
            if (z[ii]<zx):
                H[ii]= 0
            else:
                H[ii]=S0x
    
    

    return Tu,Tu_zh, T, H, nu_h, S0x, z, zx, zh, qi,qf,fI, L,kappa_par
 

def Tfront(z):
    
    T=1
    return

 

def S0x_func(L,zx,U,fI,nu):
    S0x=1/(L-zx)*(U*np.sqrt(fI)*nu*((L-zx)/2)**(2/7))**(7/5)
    return S0x

def S0x_func2(L,zx,fI,nu,Tu,kappa_par,U):
    S0x=1/(L-zx)* np.sqrt(fI)*nu*Tu*(2*kappa_par)**(4/14) *U
    return S0x
    
def qi_func(zh, S0,L,zx):
    qi=np.zeros(zh.shape[0])
    
    for ii in range(zh.shape[0]):
        qi[ii]=qi_zh(zh[ii],S0,L,zx)

    return qi

def qi_zh(zh,S0,L,zx):
    if (zh<zx):  
        qi=- S0*(L - zx)
    else:
        qi= -S0*(L - zh)

    return qi


def qf_func(kappa_par,fI,nu,U, zh, B_Bx,S0,L,zx):

    qf=np.zeros(zh.shape[0])
    for ii in range(zh.shape[0]):
        qf[ii]=qf_zh(kappa_par,fI,nu,U,zh[ii],B_Bx[ii],S0,L,zx)

    return qf

def qf_zh(kappa_par,fI,nu,U,zh,B_Bx,S0,L,zx):
    if (zh<zx):
        qf = -U *np.sqrt(fI) *nu * 1/B_Bx *(S0 *(L-zx))**(2/7) *  ((zx-zh)/3 *(1 + abs(B_Bx) +(abs(B_Bx))**2 ) + (L-zx)/2)**(2/7)
    else:
        qf = -U * np.sqrt(fI) * nu * 1/B_Bx * (S0/2)**(2/7) *(L-zh)**(4/7)
            
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
        Tu[ii]=Tupstream(So,L,zx,kappa_par,zh[ii],B_Bx[ii])

    return Tu

def Tuupstream(S0,L,zx,kappa_par,zh,B_Bx_zh):
    if (zh<zx):  
        Tu=(7*S0*( L-zx)/(2*kappa_par))**(2/7) * ((zx-zh)/3 *(1 + abs(B_Bx_zh) +(abs(B_Bx_zh))**2 ) + (L-zx)/2)**(2/7)
    else:
        Tu= (7*S0/(4*kappa_par))**(2/7) *(L-zh)**(4/7)

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
