from __future__ import print_function
import numpy as np
from openmdao.api import ExecComp, IndepVarComp, Component, Problem, Group
from openmdao.api import ScipyOptimizer
import node as nd

import matplotlib.pyplot as plt

t0 = 0
n = 20 #ノード数
div_size = n+2 #ノード数+初期条件+終端条件

#tau, w = nd.legendre_gauss(n)
#D = nd.deriv_mat(tau)
tau, w, D = nd.make_node_derivative_matrix(n)

#運動方程式
def dynamics(u, v, x, y, beta):
    (_u, _v, _x, _y, _beta) \
        = (u[1:div_size-1], v[1:div_size-1], x[1:div_size-1], y[1:div_size-1], beta[1:div_size-1])
    a = 1.0
    dx = np.zeros(0)
    dx0 = a*np.cos(_beta)
    dx1 = a*np.sin(_beta)
    dx2 = _u
    dx3 = _v
    dx = np.hstack((dx0, dx1, dx2, dx3))
    return dx

#評価関数
class tfComp(Component):
    def __init__(self):
        super(tfComp,self).__init__()
        self.add_param('t', val = 2.0)
        self.add_output('tf', val = 0.0)
        self.deriv_options['type'] = 'fd'

    def solve_nonlinear(self, params, unknowns, resids):
        t = params['t']
        unknowns['tf'] = t

#等式拘束条件
class eqComp(Component):
    def __init__(self):
        super(eqComp,self).__init__()
        self.add_param('u', val = np.zeros(div_size))
        self.add_param('v', val = np.zeros(div_size))
        self.add_param('x', val = np.zeros(div_size))
        self.add_param('y', val = np.zeros(div_size))
        self.add_param('beta', val = np.zeros(div_size))
        self.add_param('tf', val = 2.0)
        #self.add_output('con1', np.zeros((div_size-2)*4))
        #self.add_output('con2', np.zeros(div_size-1))
        #self.add_output('con3', np.zeros(div_size-1))
        #self.add_output('con4', np.zeros(div_size-1))
        #self.add_output('con5', np.zeros(div_size-1))
        #self.add_output('con6', 0.0)
        #self.add_output('con7', 0.0)
        #self.add_output('con8', 0.0)
        #self.add_output('con9', 0.0)
        #self.add_output('con10', 0.0)
        #self.add_output('con11', 0.0)
        #self.add_output('con12', 0.0)
        self.deriv_options['type'] = 'fd'
        self.add_output('eq',np.zeros((div_size-2)*4+11))

    def solve_nonlinear(self, params, unknowns, resids):
        _u = params['u']
        _v = params['v']
        _x = params['x']
        _y = params['y']
        _beta = params['beta']
        u = _u[0:div_size-1]
        v = _v[0:div_size-1]
        x = _x[0:div_size-1]
        y = _y[0:div_size-1]
        beta = _beta[0:div_size-1]
        tf = params['tf']
        dx = dynamics(u,v,x,y,beta)
        derivative = np.hstack((D.dot(u), D.dot(v), D.dot(x), D.dot(y)))
        #unknowns['con1'] = derivative - (tf-t0)/2.0*dx #配列のサイズが異なる
        #unknowns['con2'] = _u[div_size-1]-_u[0]-np.sum(D.dot(u)*w)
        #unknowns['con3'] = _v[div_size-1]-_v[0]-np.sum(D.dot(v)*w)
        #unknowns['con4'] = _x[div_size-1]-_x[0]-np.sum(D.dot(x)*w)
        #unknowns['con5'] = _y[div_size-1]-_y[0]-np.sum(D.dot(y)*w)
        #境界条件
        #unknowns['con6'] = u[0]-0.0
        #unknowns['con7'] = v[0]-0.0
        #unknowns['con8'] = x[0]-0.0
        #unknowns['con9'] = y[0]-0.0
        #unknowns['con10'] = _u[div_size-1]-1.0
        #unknowns['con11'] = _v[div_size-1]-0.0
        #unknowns['con12'] = _y[div_size-1]-1.0
        #拘束条件
        con1 = derivative - (tf - t0) / 2.0 * dx
        con2 = _u[div_size - 1] - _u[0] - np.sum(D.dot(u) * w)
        con3 = _v[div_size - 1] - _v[0] - np.sum(D.dot(v) * w)
        con4 = _x[div_size - 1] - _x[0] - np.sum(D.dot(x) * w)
        con5 = _y[div_size - 1] - _y[0] - np.sum(D.dot(y) * w)
        # 境界条件
        con6 = u[0] - 0.0
        con7 = v[0] - 0.0
        con8 = x[0] - 0.0
        con9 = y[0] - 0.0
        con10 = _u[div_size - 1] - 1.0
        con11 = _v[div_size - 1] - 0.0
        con12 = _y[div_size - 1] - 1.0
        unknowns['eq'] \
            = np.hstack((con1,con2,con3,con4,con5,con6,con7,con8,con9,con10,con11,con12))

#不等式拘束条件
class ineqComp(Component):
    def __init__(self):
        super(ineqComp,self).__init__()
        self.add_param('u', val=np.zeros(div_size))
        self.add_param('v', val=np.zeros(div_size))
        self.add_param('x', val=np.zeros(div_size))
        self.add_param('y', val=np.zeros(div_size))
        self.add_param('beta', val=np.zeros(div_size))
        self.add_param('tf', val= 2.0)
        #self.add_output('con1', np.zeros(div_size-1))
        #self.add_output('con2', np.zeros(div_size-1))
        #self.add_output('con3', np.zeros(div_size-1))
        self.deriv_options['type'] = 'fd'
        self.add_output('ineq',np.zeros((div_size-1)*2+1))

    def solve_nonlinear(self, params, unknowns, resids):
        _u = params['u']
        _v = params['v']
        _x = params['x']
        _y = params['y']
        _beta = params['beta']
        u = _u[0:div_size - 1]
        v = _v[0:div_size - 1]
        x = _x[0:div_size - 1]
        y = _y[0:div_size - 1]
        beta = _beta[0:div_size - 1]
        tf = params['tf']
        dx = dynamics(u,v,x,y,beta)
        con1 = beta+np.pi/2.0
        con2 = np.pi/2.0-beta
        con3 = _u[div_size-1]-1.0
        #unknowns['con1'] = beta + np.pi / 2.0
        #unknowns['con2'] = np.pi / 2.0 - beta
        #unknowns['con3'] = _u[div_size - 1] - 1.0
        unknowns['ineq'] = np.hstack((con1,con2,con3))

if __name__ == "__main__":
    #top = Problem()
    #root = top.root = Group()
    root = Group()

    root.add('p1',IndepVarComp('u',np.ones(div_size)*0.5))
    root.add('p2',IndepVarComp('v',np.ones(div_size)*0.5))
    root.add('p3',IndepVarComp('x',np.ones(div_size)*0.5))
    root.add('p4',IndepVarComp('y',np.ones(div_size)*0.5))
    root.add('p5',IndepVarComp('beta',np.ones(div_size)*0.5))
    root.add('p6',IndepVarComp('tf',2.3))

    root.add('peq',eqComp())
    root.add('pineq',ineqComp())

    root.add('obj', tfComp())

    root.connect('p1.u','peq.u')
    root.connect('p2.v', 'peq.v')
    root.connect('p3.x', 'peq.x')
    root.connect('p4.y', 'peq.y')
    root.connect('p5.beta', 'peq.beta')
    root.connect('p6.tf', 'peq.tf')

    root.connect('p1.u', 'pineq.u')
    root.connect('p2.v', 'pineq.v')
    root.connect('p3.x', 'pineq.x')
    root.connect('p4.y', 'pineq.y')
    root.connect('p5.beta', 'pineq.beta')
    root.connect('p6.tf', 'pineq.tf')

    root.connect('p6.tf', 'obj.t')
    root.connect('peq.tf', 'obj.t')
    root.connect('pineq.tf', 'obj.t')

    top = Problem(root)
    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'SLSQP'
    top.driver.options['maxiter'] = 250

    top.driver.add_objective('obj.tf')

    top.driver.add_desvar('p1.u', lower=0.0)
    top.driver.add_desvar('p2.v', upper=1.0)
    top.driver.add_desvar('p3.x', lower=0.0, upper=1.0)
    top.driver.add_desvar('p4.y', lower=0.0, upper=2.0)
    top.driver.add_desvar('p5.beta', lower=-np.pi/2.0, upper=np.pi/2.0)
    top.driver.add_desvar('p6.tf', lower=2.0, upper=3.0)

    top.driver.add_constraint('peq.eq',equals=np.zeros((div_size-2)*4+11))
    #top.driver.add_constraint('peq.con1',equals=np.zeros((div_size-2)*4))
    #top.driver.add_constraint('peq.con2',equals=np.zeros(div_size-1))
    #top.driver.add_constraint('peq.con3',equals=np.zeros(div_size-1))
    #top.driver.add_constraint('peq.con4',equals=np.zeros(div_size-1))
    #top.driver.add_constraint('peq.con5',equals=np.zeros(div_size-1))
    #top.driver.add_constraint('peq.con6',equals=0.0)
    #top.driver.add_constraint('peq.con7',equals=0.0)
    #top.driver.add_constraint('peq.con8',equals=0.0)
    #top.driver.add_constraint('peq.con9',equals=0.0)
    #top.driver.add_constraint('peq.con10',equals=0.0)
    #top.driver.add_constraint('peq.con11',equals=0.0)
    #top.driver.add_constraint('peq.con12',equals=0.0)
    top.driver.add_constraint('pineq.ineq',lower=np.zeros((div_size-1)*2+1))
    #top.driver.add_constraint('pineq.con1',lower=np.zeros(div_size-1))
    #top.driver.add_constraint('pineq.con2',lower=np.zeros(div_size-1))
    #top.driver.add_constraint('pineq.con3',lower=np.zeros(div_size-1))

    top.setup()

    #top['p1.u'] = np.zeros(div_size)
    #top['p2.v'] = np.zeros(div_size)
    #top['p3.x'] = np.zeros(div_size)
    #top['p4.y'] = np.zeros(div_size)
    #top['p5.beta'] = np.zeros(div_size)
    #top['p6.tf'] = -1.0

    top.run()
    result = top['obj.tf']
    print(result)

    u = top['p1.u']
    v = top['p2.v']
    x = top['p3.x']
    y = top['p4.y']
    beta = top['p5.beta']
    tf = top['p6.tf']
    #tf = result
    time = (tf - t0) / 2.0 * tau + (tf + t0) / 2.0
    print(time)

    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(time,u,marker = 'o',label=u"vel x")
    plt.plot(time,v,marker = 'o',label=u"vel y")
    plt.legend(loc="best")
    plt.ylabel(u"velocity [m/s]")
    plt.subplot(4,1,2)
    plt.plot(time,x,marker = 'o',label=u"x")
    plt.plot(time,y,marker = 'o',label=u"y")
    plt.legend(loc="best")
    plt.ylabel(u"position [m]")
    plt.subplot(4,1,3)
    plt.plot(time,beta,marker = 'o',label=u"beta")
    plt.legend(loc="best")
    plt.ylabel(u"angle [rad]")
    plt.xlabel(u"time [s]")
    plt.subplot(4,1,4)
    plt.plot(x,y,marker = 'o')
    plt.xlabel(u"position x[m]")
    plt.ylabel(u"position y[m]")
    plt.show()
