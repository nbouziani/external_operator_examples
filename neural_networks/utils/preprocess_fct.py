from firedrake import *
import numpy as np

# Convert a firedrake.Function into a matrix numpy array whose entries correspond to evaluation of the Function at the dofs

def convert_func_(fct, P):
    #fct = Function(P).interpolate(u)
    mesh = P.mesh()
    x, y = SpatialCoordinate(mesh)
    fx = Function(P).interpolate(x)
    fy = Function(P).interpolate(y)
    def grid_data(vx, vy): 
        res = {} 
        for i, e in enumerate(vy): 
            if e in res.keys(): 
                res[e].append(i) 
            else: 
                res[e] = [i] 
        sk = sorted(res.keys(), reverse=True) 
        p = () 
        for e in sk: 
            rr = sorted(res[e], key=lambda c:vx[c]) 
            p += tuple(rr) 
        return p 
    pr = grid_data(fx.dat.data, fy.dat.data) 
    res = np.array(list(fct.dat.data[i] for i in pr)) 
    #real_res = np.real(res).reshape(256,256) 
    #real_im = np.imag(res).reshape(256,256)    
  
    return res.reshape(256,256)

def revert_func_(u_np, P):
    mesh = P.mesh()
    x, y = SpatialCoordinate(mesh)
    fx = Function(P).interpolate(x)
    fy = Function(P).interpolate(y)
    def grid_data(vx, vy):
        res = {}
        for i, e in enumerate(vy):
            if e in res.keys():
                res[e].append(i)
            else:
                res[e] = [i]
        sk = sorted(res.keys(), reverse=True)
        p = ()
        for e in sk:
            rr = sorted(res[e], key=lambda c:vx[c])
            p += tuple(rr)
        return p
    pr = grid_data(fx.dat.data, fy.dat.data)
    u_np = u_np.flatten()
    res = np.zeros(len(pr))
    for i, e in enumerate(pr):
        res[e] = u_np[i]
    u = Function(P)
    u.dat.data[:] = res
    return u
