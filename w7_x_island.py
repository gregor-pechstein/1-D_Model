from osa import Client
import numpy as np
import matplotlib.pyplot as plt

def PoincarePlot(config,tracer,machine,pos,numPoints=300,phi=0, plotPoin=False):
    
    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints =numPoints
    poincare.phi0 = [phi]

    task = tracer.types.Task()
    task.step = 0.2
    task.poincare = poincare
    
    res = tracer.service.trace(pos, config, task,machine , None)
    


    if (plotPoin==True):    
        path="/home/grepeloc/Simulation/1-D_Model/"
        plt.rc('font', family='Serif')
        plt.figure() #figsize=(8,4.5))
        for i in range(0, len(res.surfs)):
            c=np.sqrt(np.array(res.surfs[i].points.x1)**2+np.array(res.surfs[i].points.x2)**2)
            plt.scatter(c, res.surfs[i].points.x3, color="red", s=0.1)
            #plt.scatter(res.surfs[i].points.x1, res.surfs[i].points.x3, color="red", s=0.1)
            plt.axis('equal')
        #plt.savefig(path+'/Poincare'+str(phi)+'.png', dpi=300)
        plt.show()

    return res




def LineTracing(config,tracer,pos,numsteps,steplen=0.05):
    
    lineTask = tracer.types.LineTracing()
    lineTask.numSteps = numsteps
    
    task = tracer.types.Task()
    task.step = steplen
    task.lines = lineTask
    
    res = tracer.service.trace(pos, config, task, None, None)


    return res

def Separatrix(config,tracer):
    # finding the Separatrix
    settings = tracer.types.SeparatrixSettings()
    settings.axisSettings = tracer.types.AxisSettings()
    
    Sep = tracer.service.findSeparatrix(0.05, config, settings)

    sep_pos = tracer.types.Points3D()
    sep_pos.x1 = Sep.x
    sep_pos.x2 = Sep.y
    sep_pos.x3 = Sep.z

    return Sep, sep_pos

def XP(config,tracer,Sep):
    #using a point on the Separatrix as Guess for the findXpoint func 
    xp = tracer.types.XPointGuess()
    xp.xpointX = Sep.x
    xp.xpointY = Sep.y
    xp.xpointZ = Sep.z
    xp.separatrixX = Sep.x
    xp.separatrixY = Sep.y
    xp.separatrixZ = Sep.z

    
    settings = tracer.types.SeparatrixSettings()
    settings.axisSettings = tracer.types.AxisSettings()
    ''' make a request to the web service: '''
    
    XPoint = tracer.service.findXPoints(0.05,xp,config,settings) #step, guess, config,settings

    return XPoint

def MagCharacteristic(config,tracer,p):
    task = tracer.types.Task()
    task.step = 0.01
     
    task.characteristics = tracer.types.MagneticCharacteristics()
    task.characteristics.axisSettings = tracer.types.AxisSettings()
    
    res = tracer.service.trace(p, config, task, None,  None)

    return res

def Magneticfield(config,tracer,line):
    B = tracer.service.magneticField(line, config)

    B_abs=np.zeros(len(B.field.x1))
    for ii in range(0,len(B.field.x1)):
        B_abs[ii]=np.sqrt( B.field.x1[ii]**2+ B.field.x2[ii]**2+ B.field.x3[ii]**2)
    return B,B_abs

def Conection_lenght(config,tracer,machine,points,limit=5000):
    task = tracer.types.Task()
    task.step = 6.5e-3
    con = tracer.types.ConnectionLength()
    con.limit = limit
    con.returnLoads = False
    task.connection = con


    config.inverseField = False

    res_fwd = tracer.service.trace(points, config, task, machine)

    config.inverseField = True

    res_bwd = tracer.service.trace(points, config, task, machine)


    return res_fwd, res_bwd


def B_connection_allong_line():
    path="/home/grepeloc/Simulation/1-D_Model/"

    machine, config,tracer=Grid()

    Sep,sep_pos = Separatrix(config,tracer)

    XPoint      = XP(config,tracer,Sep)

    x_start = tracer.types.Points3D()
    x_start.x1 = XPoint.xpoints.x1[2]
    x_start.x2 = XPoint.xpoints.x2[2]
    x_start.x3 = XPoint.xpoints.x3[2]
    
    
    XLine       = LineTracing(config,tracer,x_start,727)

    MagChara_XLine = MagCharacteristic(config,tracer,XLine.lines[0].vertices)
    phi=np.zeros(len(MagChara_XLine.characteristics))
    theta=np.zeros(len(MagChara_XLine.characteristics))
    reff=np.zeros(len(MagChara_XLine.characteristics))
    for ii in range(0, len(MagChara_XLine.characteristics)):
        phi[ii]= MagChara_XLine.characteristics[ii].phi0
        theta[ii]= MagChara_XLine.characteristics[ii].theta0
        reff[ii]= MagChara_XLine.characteristics[ii].reff
        


    ######taking a point in the island( a small stepp away from the Xpoint to trace the fildline in the island"
    Island_pos = tracer.types.Points3D()
    Island_pos.x1 = XPoint.xpoints.x1[2]+0.01
    Island_pos.x2 = XPoint.xpoints.x2[2]
    Island_pos.x3 = XPoint.xpoints.x3[2]+0.01

    
    Poin_sep = PoincarePlot(config,tracer,machine,sep_pos)
    Poin_Island = PoincarePlot(config,tracer,machine,Island_pos,numPoints=1000,phi=0) 

    ###tracing a fildline in the island 
    IslandTrace=  LineTracing(config,tracer,Island_pos,12000)

    ###tracing Poincare point on seperatrix
    SepTrace   =  LineTracing(config,tracer,Poin_sep.surfs[0].points,727)


    B,B_abs = Magneticfield(config,tracer,IslandTrace.lines[0].vertices)

    B_x,Bx_abs = Magneticfield(config,tracer,XPoint.xpoints)


    #Conlengh_x_fwd, Conlengh_x_bwd= Conection_lenght(config,tracer,machine,x_start)
    Conlengh_x_fwd, Conlengh_x_bwd= Conection_lenght(config,tracer,machine,Island_pos)
    Conlengh_I_fwd, Conlengh_I_bwd =Conection_lenght(config,tracer,machine,IslandTrace.lines[0].vertices)

    length_fwd= np.zeros(len(Conlengh_I_fwd.connection))
    length_bwd= np.zeros(len(Conlengh_I_bwd.connection))
    for i in range(0, len(Conlengh_I_fwd.connection)):
        length_fwd[i]= Conlengh_I_fwd.connection[i].length
        length_bwd[i]=Conlengh_I_bwd.connection[i].length

    idx = np.argwhere(np.diff(np.sign(length_bwd - length_fwd))).flatten()
    L=length_bwd[idx[0]]
    zx= length_bwd[0]

    figurePloting(Poin_Island,Poin_sep,XPoint,SepTrace,XLine,IslandTrace,B_abs,Bx_abs,lineTask_x, lineTask_Island,path)
    
        
    return Sep,sep_pos,B_x,Bx_abs, B,B_abs,Conlengh_x_fwd, Conlengh_x_bwd,Conlengh_I_fwd, Conlengh_I_bwd, XPoint, XLine,SepTrace, traceLengh_x,traceLengh,Poin_Island,IslandTrace



def figurePloting(Poin_Island,Poin_sep,XPoint,SepTrace,XLine,IslandTrace,B_abs,Bx_abs,lineTask_x, lineTask_Island,path):
    

    ''' plot the points: '''
    plt.rc('font', family='Serif')
    plt.figure() #figsize=(8,4.5))
    for i in range(0, len(Poin_sep.surfs)):
        plt.scatter(Poin_sep.surfs[i].points.x1, Poin_sep.surfs[i].points.x3, color="red", s=0.1)
    alpha=np.linspace(0.1,1,len(Poin_Island.surfs[0].points.x1))
    for i in range(0,len(Poin_Island.surfs[0].points.x1) ):
        plt.scatter(Poin_Island.surfs[0].points.x1[i], Poin_Island.surfs[0].points.x3[i],marker='x', color="red",alpha=alpha[i])
    plt.plot(XPoint.xpoints.x1[1], XPoint.xpoints.x3[1], marker='.',label=r'Xpoint',color ="black")    
    plt.plot(XPoint.xpoints.x1[2], XPoint.xpoints.x3[2], marker='.',color ="black")
    plt.axis('equal')
    plt.grid(alpha=0.5)
    plt.xlabel(r'R [m]', fontsize=18)
    plt.ylabel(r'$Z [m]$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
    plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/Poincare.png', dpi=300)
    plt.show()

    # plt.rc('font', family='Serif')
    # plt.figure()
    # plt.axis([-7.0, 7.0, -5.5, 5.5]) 
    #  for i in range(0, len(SepTrace.lines)):
    #     plt.scatter(SepTrace.lines[i].vertices.x1, SepTrace.lines[i].vertices.x2,color="red", s=0.01,alpha=0.3)
    # for i in range(0, len(XLine.lines)):
    #     plt.scatter(XLine.lines[i].vertices.x1, XLine.lines[i].vertices.x2,color ="black", s=1)
    # plt.scatter(IslandTrace.lines[i].vertices.x1, IslandTrace.lines[i].vertices.x3,color ="blue", s=1)
    # plt.axis('equal')
    # plt.grid(alpha=0.5)
    # plt.xlabel(r'R [m]', fontsize=18)
    # plt.ylabel(r'$Z [m]$', fontsize=18)
    # plt.tick_params('both', labelsize=14)
    # plt.tight_layout()
    ## plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    # plt.savefig(path+'/XLine.png', dpi=300)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(0, len(SepTrace.lines)):
        ax.scatter(SepTrace.lines[i].vertices.x1, SepTrace.lines[i].vertices.x2, SepTrace.lines[i].vertices.x3,color="red", s=0.01,alpha=0.3)
    for i in range(0, len(XLine.lines)):
        ax.scatter(XLine.lines[i].vertices.x1,XLine.lines[i].vertices.x2, XLine.lines[i].vertices.x3,color ="black", s=1)
    for i in range(0, len(IslandTrace.lines)):
        ax.scatter(IslandTrace.lines[i].vertices.x1,IslandTrace.lines[i].vertices.x2, IslandTrace.lines[i].vertices.x3,color ="green", s=1)

    plt.show()
                    

    traceLengh=np.arange(0, (lineTask_Island.numSteps+1) * task_Island.step,task_Island.step)
    plt.rc('font', family='Serif')
    plt.figure()
    plt.plot(traceLengh,B_abs)
    plt.grid(alpha=0.5)
    plt.ylabel(r'B [T]', fontsize=18)
    plt.xlabel(r'$L [m]$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
   ## plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/B_absIsland.png', dpi=300)
    plt.show()

    traceLengh_x=np.arange(0, (lineTask_x.numSteps+1) * task_x.step,task_x.step)
    plt.rc('font', family='Serif')
    plt.figure()
    plt.plot(traceLengh_x,Bx_abs)
    plt.grid(alpha=0.5)
    plt.ylabel(r'B [T]', fontsize=18)
    plt.xlabel(r'$L [m]$', fontsize=18)
    plt.tick_params('both', labelsize=14)
    plt.tight_layout()
   ## plt.legend(fancybox=True, loc='lower right', framealpha=0, fontsize=12)
    plt.savefig(path+'/Bx_abs.png', dpi=300)
    plt.show()


    return







def Grid():

    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')
    
    config = tracer.types.MagneticConfig()
    config.configIds = [1]  ## Just use machine IDs instead of coil currents because it's easier.

    
    grid = tracer.types.Grid()
    grid.fieldSymmetry = 5

    cyl = tracer.types.CylindricalGrid()
    cyl.numR, cyl.numZ, cyl.numPhi = 181, 181, 481
    cyl.RMin, cyl.RMax, cyl.ZMin, cyl.ZMax = 4.05, 6.75, -1.35, 1.35
    grid.cylindrical = cyl

    config.grid = grid

    machine = tracer.types.Machine(1)
    machine.grid.numX, machine.grid.numY, machine.grid.numZ = 500,500,100
    machine.grid.ZMin, machine.grid.ZMax = -1.5,1.5
    machine.grid.YMin, machine.grid.YMax = -7, 7
    machine.grid.XMin, machine.grid.XMax = -7, 7
    #machine.meshedModelsIds = [164]
    machine.assemblyIds = [2]

    return machine, config,tracer
