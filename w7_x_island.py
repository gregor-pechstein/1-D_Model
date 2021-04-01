from osa import Client
import numpy as np
import matplotlib.pyplot as plt

def PoincarePlot():
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    pos = tracer.types.Points3D()
    pos.x1 = np.linspace(5.64, 6.3, 30)
    pos.x2 = np.zeros(30)
    pos.x3 = np.zeros(30)

    config = tracer.types.MagneticConfig()
    config.configIds = [1]

    my_grid = tracer.types.CylindricalGrid()
    my_grid.RMin = 4.05
    my_grid.RMax = 6.75
    my_grid.ZMin = -1.35
    my_grid.ZMax = 1.35
    my_grid.numR = 181
    my_grid.numZ = 181
    my_grid.numPhi = 481
    
    g = tracer.types.Grid()
    g.cylindrical = my_grid
    g.fieldSymmetry = 5
    
    config.grid = g



    machine = tracer.types.Machine()
    machine.meshedModelsIds = [165] 
    machine_grid = tracer.types.CartesianGrid()
    machine_grid.XMin = -7
    machine_grid.XMax = 7
    machine_grid.YMin = -7
    machine_grid.YMax = 7
    machine_grid.ZMin = -1.5
    machine_grid.ZMax = 1.5
    machine_grid.numX = 400
    machine_grid.numY = 400
    machine_grid.numZ = 100

    machine.grid = machine_grid

    
    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints =300
    poincare.phi0 = [0.]

    task = tracer.types.Task()
    task.step = 0.2
    task.poincare = poincare
    
    res = tracer.service.trace(pos, config, task,machine , None)
    
    ''' number of PoincareSurface objets: 80'''
    print(len(res.surfs))

    ''' plot the points: '''
    for i in range(0, len(res.surfs)):
        plt.scatter(res.surfs[i].points.x1, res.surfs[i].points.x3, color="red", s=0.1)
    plt.show()

    return




def LineTracing():
    
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')
    
    pos = tracer.types.Points3D()
    pos.x1 = np.linspace(6.1, 6.3, 30)
    pos.x2 = np.zeros(30)
    pos.x3 = np.zeros(30)
    
    config = tracer.types.MagneticConfig()
    config.configIds = [1]

    my_grid = tracer.types.CylindricalGrid()
    my_grid.RMin = 4.05
    my_grid.RMax = 6.75
    my_grid.ZMin = -1.35
    my_grid.ZMax = 1.35
    my_grid.numR = 181
    my_grid.numZ = 181
    my_grid.numPhi = 481
    
    g = tracer.types.Grid()
    g.cylindrical = my_grid
    g.fieldSymmetry = 5
    
    config.grid = g

    machine = tracer.types.Machine()
    machine.meshedModelsIds = [165] 
    machine_grid = tracer.types.CartesianGrid()
    machine_grid.XMin = -7
    machine_grid.XMax = 7
    machine_grid.YMin = -7
    machine_grid.YMax = 7
    machine_grid.ZMin = -1.5
    machine_grid.ZMax = 1.5
    machine_grid.numX = 400
    machine_grid.numY = 400
    machine_grid.numZ = 100

    machine.grid = machine_grid
    
    lineTask = tracer.types.LineTracing()
    lineTask.numSteps = 2000
    
    task = tracer.types.Task()
    task.step = 0.05
    task.lines = lineTask
    
    res = tracer.service.trace(pos, config, task, None, None)
    
    plt.axis([-7.0, 7.0, -5.5, 5.5])
    #fig = plt.figure()
   # ax = fig.add_subplot(projection='3d')
    for i in range(0, len(res.lines)):
        plt.scatter(res.lines[i].vertices.x1, res.lines[i].vertices.x3, s=0.01)
    plt.show()

    return



def B_connection_allong_line():
    path="/home/grepeloc/Simulation/1-D_Model/"
    
    tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')
    
    config = tracer.types.MagneticConfig()
    config.configIds = [1]

    my_grid = tracer.types.CylindricalGrid()
    my_grid.RMin = 4.05
    my_grid.RMax = 6.75
    my_grid.ZMin = -1.35
    my_grid.ZMax = 1.35
    my_grid.numR = 181
    my_grid.numZ = 181
    my_grid.numPhi = 481
    
    g = tracer.types.Grid()
    g.cylindrical = my_grid
    g.fieldSymmetry = 5
    
    config.grid = g

    machine = tracer.types.Machine()
    machine.meshedModelsIds = [165] 
    machine_grid = tracer.types.CartesianGrid()
    machine_grid.XMin = -7
    machine_grid.XMax = 7
    machine_grid.YMin = -7
    machine_grid.YMax = 7
    machine_grid.ZMin = -1.5
    machine_grid.ZMax = 1.5
    machine_grid.numX = 400
    machine_grid.numY = 400
    machine_grid.numZ = 100

    machine.grid = machine_grid

    # finding the Separatrix
    settings = tracer.types.SeparatrixSettings()
    settings.axisSettings = tracer.types.AxisSettings()
    
    Sep = tracer.service.findSeparatrix(0.05, config, settings)

    sep_pos = tracer.types.Points3D()
    sep_pos.x1 = Sep.x
    sep_pos.x2 = Sep.y
    sep_pos.x3 = Sep.z

    #using a point on the Separatrix as Guess for the findXpoint func 
    xp = tracer.types.XPointGuess()
    xp.xpointX = Sep.x
    xp.xpointY = Sep.y
    xp.xpointZ = Sep.z
    xp.separatrixX = Sep.x
    xp.separatrixY = Sep.y
    xp.separatrixZ = Sep.z

    #xp.xpointX = 5.45
    #xp.xpointY = 0
    #xp.xpointZ = 0.88
    #xp.separatrixX = 5.45
    #xp.separatrixY = 0
    #xp.separatrixZ = 0.88
    
    settings = tracer.types.SeparatrixSettings()
    settings.axisSettings = tracer.types.AxisSettings()
    ''' make a request to the web service: '''
    
    XPoint = tracer.service.findXPoints(0.05,xp,config,settings) #step, guess, config,settings

    Island_pos = tracer.types.Points3D()
    Island_pos.x1 = XPoint.xpoints.x1[2]+0.01
    Island_pos.x2 = XPoint.xpoints.x2[2]
    Island_pos.x3 = XPoint.xpoints.x3[2]+0.01
    #Island_pos.x1 = np.arange(XPoint.xpoints.x1[2]+0.01,XPoint.xpoints.x1[2]+0.05,0.005)
    #Island_pos.x2 =np.full(Island_pos.x1.shape[0], XPoint.xpoints.x2[2])
    #Island_pos.x3 = np.arange(XPoint.xpoints.x3[2]+0.01,XPoint.xpoints.x3[2]+0.05,0.005)
    
    ''' plot the results: '''

    
   

    #poincare plot Separatrix
    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints =300
    poincare.phi0 = [0.]

    task = tracer.types.Task()
    task.step = 0.2
    task.poincare = poincare
    #Island
    poincareIsl = tracer.types.PoincareInPhiPlane()
    poincareIsl.numPoints =10000
    poincareIsl.phi0 = [0.]

    taskIsl = tracer.types.Task()
    taskIsl.step = 0.2
    taskIsl.poincare = poincare
    
    Poin_sep = tracer.service.trace(sep_pos, config, task,machine , None)
    Poin_Island = tracer.service.trace(Island_pos, config, taskIsl,machine , None)
    
    ''' number of PoincareSurface objets: 80'''
    print(len(Poin_sep.surfs))

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
    
    lineTask_Island = tracer.types.LineTracing()
    lineTask_Island.numSteps = 12000
    
    task_Island = tracer.types.Task()
    task_Island.step = 0.05
    task_Island.lines = lineTask_Island
    
    lineTask_Sep = tracer.types.LineTracing()
    lineTask_Sep.numSteps = 727
    
    task_Sep = tracer.types.Task()
    task_Sep.step = 0.05
    task_Sep.lines = lineTask_Sep

    lineTask_x = tracer.types.LineTracing()
    lineTask_x.numSteps = 727
    
    task_x = tracer.types.Task()
    task_x.step = 0.05
    task_x.lines = lineTask_x

    x_start = tracer.types.Points3D()
    x_start.x1 = XPoint.xpoints.x1[2]
    x_start.x2 = XPoint.xpoints.x2[2]
    x_start.x3 = XPoint.xpoints.x3[2]

    #FieldLine= tracer.service.trace(sep_pos, config, task, None, None)
    IslandTrace= tracer.service.trace(Island_pos, config, task_Island, None, None)
    XLine= tracer.service.trace(x_start, config, task_x, None, None)
    SepTrace= tracer.service.trace(Poin_sep.surfs[0].points, config, task_Sep, None, None)


   # plt.rc('font', family='Serif')
   # plt.figure()
   # plt.axis([-7.0, 7.0, -5.5, 5.5])

 
   # for i in range(0, len(SepTrace.lines)):
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
                    

    B = tracer.service.magneticField(IslandTrace.lines[0].vertices, config)

    B_abs=np.zeros(len(B.field.x1))
    for ii in range(0,len(B.field.x1)):
        B_abs[ii]=np.sqrt( B.field.x1[ii]**2+ B.field.x2[ii]**2+ B.field.x3[ii]**2)

    B_x = tracer.service.magneticField(XLine.lines[0].vertices, config)

    Bx_abs=np.zeros(len(B_x.field.x1))
    for ii in range(0,len(B_x.field.x1)):
        Bx_abs[ii]=np.sqrt( B_x.field.x1[ii]**2+ B_x.field.x2[ii]**2+ B_x.field.x3[ii]**2)

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

    task = tracer.types.Task()
    task.step = 6.5e-3
    con = tracer.types.ConnectionLength()
    con.limit = 2.0e4
    con.returnLoads = True
    task.connection = con

    Conlengh_x = tracer.service.trace(x_start, config, task, machine)
    Conlengh_Island = tracer.service.trace(Island_pos, config, task, machine)
    
    return Sep,sep_pos,B_x,Bx_abs, B,B_abs,Conlengh_x, Conlengh_Island, XPoint, XLine,SepTrace, traceLengh_x,traceLengh,Poin_Island,IslandTrace





def Conection_lenght(x,y,z):
    tracer = Client('http://esb:8280/services/FieldLineProxy?wsdl')
    points = tracer.types.Points3D()
    points.x1 = x
    points.x2 = y
    points.x3 = z


    ### copied from webservices...#
    task = tracer.types.Task()
    task.step = 6.5e-3
    con = tracer.types.ConnectionLength()
    con.limit = limit
    con.returnLoads = False
    task.connection = con

    # diff = tracer.types.LineDiffusion()
    # diff.diffusionCoeff = .00
    # diff.freePath = 0.1
    # diff.velocity = 5e4
    # task.diffusion = diff

    config = tracer.types.MagneticConfig()

    config.configIds = configuration  ## Just use machine IDs instead of coil currents because it's easier.

    ### This bit doesn't work when called as a function.  
    # config.coilsIds = range(160,230)
    # config.coilsIdsCurrents = [1.43e6,1.43e6,1.43e6,1.43e6,1.43e6]*10
    # # config.coilsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)


    # # config.coilsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)
    # config.coilsIdsCurrents = [1.43e6,1.43e6,1.43e6,1.43e6,1.43e6]*10
    # # config.coilsIdsCurrents.extend([0.13*1.43e6,0.13*1.43e6]*10)
    
    grid = tracer.types.Grid()
    grid.fieldSymmetry = 5

    cyl = tracer.types.CylindricalGrid()
    cyl.numR, cyl.numZ, cyl.numPhi = 181, 181, 481
    cyl.RMin, cyl.RMax, cyl.ZMin, cyl.ZMax = 4.05, 6.75, -1.35, 1.35
    grid.cylindrical = cyl

    machine = tracer.types.Machine(1)
    machine.grid.numX, machine.grid.numY, machine.grid.numZ = 500,500,100
    machine.grid.ZMin, machine.grid.ZMax = -1.5,1.5
    machine.grid.YMin, machine.grid.YMax = -7, 7
    machine.grid.XMin, machine.grid.XMax = -7, 7
    # machine.meshedModelsIds = [164]
    machine.assemblyIds = [12,14,8,9,13,21]

    config.grid = grid

    config.inverseField = False

    res_fwd = tracer.service.trace(points, config, task, machine)

    config.inverseField = True

    res_bwd = tracer.service.trace(points, config, task, machine)


    ###### end of copied code #######
