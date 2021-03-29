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


def xPoint():
    
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
    machine.meshedModelsIds = [164] 
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
    
    settings = tracer.types.AxisSettings()
    res = tracer.service.findAxis(0.05, config, settings)
    
    plt.axis([-6.0, 6.5, -5.0, 5.0])
    plt.plot(res.axis.vertices.x1, res.axis.vertices.x3)
    plt.show()

    XPointsettings = tracer.types.XPointSettings()
    res = tracer.service.findXpoints(0.05, config,XPointsettings, settings)

    plt.axis([-6.0, 6.5, -5.0, 5.0])
    plt.plot(res.axis.vertices.x1, res.axis.vertices.x3)
    plt.show()

    return




def B_connection_allong_line():
    
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
    
    settings = tracer.types.SeparatrixSettings()
    settings.axisSettings = tracer.types.AxisSettings()
    
    Sep = tracer.service.findSeparatrix(0.05, config, settings)
    
    pos = tracer.types.Points3D()
    pos.x1 = Sep.x
    pos.x2 = Sep.y
    pos.x3 = Sep.z

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
    
    ''' plot the results: '''
    
    plt.plot(XPoint.xpoints.x1, XPoint.xpoints.x3, '.')

    #poincare plot
    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints =300
    poincare.phi0 = [0.]

    task = tracer.types.Task()
    task.step = 0.2
    task.poincare = poincare
    
    Poin = tracer.service.trace(pos, config, task,machine , None)
    
    ''' number of PoincareSurface objets: 80'''
    print(len(Poin.surfs))

    ''' plot the points: '''
    for i in range(0, len(Poin.surfs)):
        plt.scatter(Poin.surfs[i].points.x1, Poin.surfs[i].points.x3, color="red", s=0.1)
    plt.show()
    

    lineTask = tracer.types.LineTracing()
    lineTask.numSteps = 2000
    
    task = tracer.types.Task()
    task.step = 0.05
    task.lines = lineTask
    
    FieldLine= tracer.service.trace(pos, config, task, None, None)
    
    plt.axis([-7.0, 7.0, -5.5, 5.5])

    for i in range(0, len(FieldLine.lines)):
        plt.scatter(FieldLine.lines[i].vertices.x1, FieldLine.lines[i].vertices.x3, s=0.01)
    plt.show()

    B = tracer.service.magneticField(FieldLine.lines[0].vertices, config)

    B_abs=np.zeros(len(B.field.x1))
    for ii in range(0,len(B.field.x1)):
        B_abs[ii]=np.sqrt( B.field.x1[ii]**2+ B.field.x2[ii]**2+ B.field.x3[ii]**2)

    plt.plot(B_abs)
    plt.show()

    task = tracer.types.Task()
    task.step = 6.5e-3
    con = tracer.types.ConnectionLength()
    con.limit = 2.0e4
    con.returnLoads = True
    task.connection = con

    res_fwd = tracer.service.trace(pos, config, task, machine)
    
    return Sep,pos,FieldLine, B,B_abs,res_fwd, XPoint

