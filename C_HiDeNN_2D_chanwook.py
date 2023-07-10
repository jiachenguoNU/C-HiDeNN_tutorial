import numpy as onp
from scipy.sparse import csc_matrix
# from scipy.sparse import  csc_matrix
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
import time
from itertools import combinations
import math
from functools import partial
import os,sys
import humanize,psutil,GPUtil
import pandas as pd
from read_mesh import read_mesh_ABQ
import matplotlib.pyplot as plt
# from matplotlib import cm

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this

# memory report
def mem_report(num, gpu_idx):
    print(f"-{num}-CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
    
    GPUs = GPUtil.getGPUs()
    gpu = GPUs[gpu_idx]
    # for i, gpu in enumerate(GPUs):
    print('---GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%\n'.format(gpu_idx, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

# from read_mesh import read_mesh_ABQ

# from src.fem.generate_mesh import box_mesh, cylinder_mesh, global_args
# from GaussSet import GaussSet

from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=4)

############# FEM functions ############
def GaussSet(Gauss_Num = 2, cuda=False):
    if Gauss_Num == 2:
        Gauss_Weight1D = [1, 1]
        Gauss_Point1D = [-1/np.sqrt(3), 1/np.sqrt(3)]
       
    elif Gauss_Num == 3:
        Gauss_Weight1D = [0.55555556, 0.88888889, 0.55555556]
        Gauss_Point1D = [-0.7745966, 0, 0.7745966]
       
        
    elif Gauss_Num == 4:
        Gauss_Weight1D = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
        Gauss_Point1D = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]

    elif Gauss_Num == 6: # double checked, 16 digits
        Gauss_Weight1D = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 
                          0.4679139345726910, 0.3607615730481386, 0.1713244923791704]
        Gauss_Point1D = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 
                         0.2386191860831969, 0.6612093864662645, 0.9324695142031521]

       
    elif Gauss_Num == 8: # double checked, 20 digits
        Gauss_Weight1D=[0.10122853629037625915, 0.22238103445337447054, 0.31370664587788728733, 0.36268378337836198296,
                        0.36268378337836198296, 0.31370664587788728733, 0.22238103445337447054,0.10122853629037625915]
        Gauss_Point1D=[-0.960289856497536231684, -0.796666477413626739592,-0.525532409916328985818, -0.183434642495649804939,
                        0.183434642495649804939,  0.525532409916328985818, 0.796666477413626739592,  0.960289856497536231684]
        
    elif Gauss_Num == 10:
        Gauss_Weight1D=[0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
                        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]
        Gauss_Point1D=[-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,  
                        0.1488743389816312,  0.4333953941292472,  0.6794095682990244,  0.8650633666889845,  0.9739065285171717]
        
    elif Gauss_Num == 20:
        Gauss_Weight1D=[0.017614007, 0.04060143, 0.062672048, 0.083276742,0.10193012, 0.118194532,0.131688638,
                        0.142096109, 0.149172986, 0.152753387,0.152753387,0.149172986, 0.142096109, 0.131688638,
                        0.118194532,0.10193012, 0.083276742,0.062672048,0.04060143,0.017614007]
            
        Gauss_Point1D=[-0.993128599, -0.963971927, -0.912234428, -0.839116972, -0.746331906, -0.636053681,
                        -0.510867002, -0.373706089, -0.227785851, -0.076526521, 0.076526521, 0.227785851,
                        0.373706089, 0.510867002, 0.636053681, 0.746331906, 0.839116972, 0.912234428, 0.963971927, 0.993128599]
    
    return Gauss_Weight1D, Gauss_Point1D

def get_quad_points(Gauss_Num, dim, elem_type):
    # This function is compatible with FEM and CFEM
    
    
    
    if dim == 1: # surface integral, 2D -> 1D
        Gauss_Weight1D, Gauss_Point1D = GaussSet(Gauss_Num_tr) # as general as possible
        quad_points = np.expand_dims(np.array(Gauss_Point1D), axis=1) # 
        quad_weights = np.array(Gauss_Weight1D) # 
        # print('quad points weights')
        # print(quad_points, quad_weights)
        
    elif dim == 2:
        
        # Quad elements
        if elem_type == 'CPE4' or elem_type == 'CPE8':
            Gauss_Weight1D, Gauss_Point1D = GaussSet(Gauss_Num)
            quad_points, quad_weights = [], []
        
            for ipoint, iweight in zip(Gauss_Point1D, Gauss_Weight1D):
                if dim < 2:
                    quad_points.append([ipoint])
                    quad_weights.append(iweight)
                else:
                    for jpoint, jweight in zip(Gauss_Point1D, Gauss_Weight1D):
                        if dim < 3:
                            quad_points.append([ipoint, jpoint])
                            quad_weights.append(iweight * jweight)
                        else:
                            for kpoint, kweight in zip(Gauss_Point1D, Gauss_Weight1D):
                                quad_points.append([ipoint, jpoint, kpoint])
                                quad_weights.append(iweight * jweight * kweight)
                       
            quad_points = np.array(quad_points) # (quad_degree*dim, dim)
            quad_weights = np.array(quad_weights) # (quad_degree,)
            
        # Triangular elements
        elif elem_type == 'CPE3' or elem_type == 'CPE6':
            if Gauss_Num == 1:
                quad_weights = np.array([1.000000000000000])
                quad_points = np.array([[0.333333333333333, 0.333333333333333]])
            
            elif Gauss_Num == 3:
                quad_weights = np.array([0.333333333333333, 0.333333333333333, 0.333333333333333])
                quad_points = np.array([[0.666666666666667, 0.166666666666667],
                                        [0.166666666666667, 0.666666666666667],
                                        [0.166666666666667, 0.166666666666667]])
               
            elif Gauss_Num == 4:
                quad_weights= np.array([-0.562500000000000, 0.520833333333333, 0.520833333333333, 0.520833333333333])
                quad_points = np.array([[0.333333333333333, 0.333333333333333],
                                        [0.600000000000000, 0.200000000000000],
                                        [0.200000000000000, 0.600000000000000],
                                        [0.200000000000000, 0.200000000000000]])

            elif Gauss_Num == 6:
                quad_weights = np.array([0.109951743655322, 0.109951743655322, 0.109951743655322, 
                                         0.223381589678011, 0.223381589678011, 0.223381589678011])
                quad_points = np.array([[0.816847572980459, 0.091576213509771],
                                        [0.091576213509771, 0.816847572980459],
                                        [0.091576213509771, 0.091576213509771],
                                        [0.108103018168070, 0.445948490915965],
                                        [0.445948490915965, 0.108103018168070],
                                        [0.445948490915965, 0.445948490915965]])
            
    return quad_points, quad_weights


def get_shape_val_functions(elem_type, dim):
    """Hard-coded first order shape functions in the parent domain.
    Important: f1-f8 order must match "self.cells" by gmsh file!
    """
    if elem_type == 'CPE4' and dim == 2:
        f1 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1)) # (-1, -1)
        f2 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: 1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: 1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1)) # (-1,  1)
        shape_fun = [f1, f2, f3, f4]
        
    elif elem_type == 'CPE8' and dim == 2:
        f1 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*(-1))*(1 - x[0]*(-1) - x[1]*(-1)) # (-1, -1)
        f2 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*(-1))*(1 - x[0]*( 1) - x[1]*(-1)) # ( 1, -1)
        f3 = lambda x: -1./4.*(1 + x[0]*( 1))*(1 + x[1]*( 1))*(1 - x[0]*( 1) - x[1]*( 1)) # ( 1,  1)
        f4 = lambda x: -1./4.*(1 + x[0]*(-1))*(1 + x[1]*( 1))*(1 - x[0]*(-1) - x[1]*( 1)) # (-1,  1)
        f5 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*(-1)) # ( 0, -1)
        f6 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*( 1)) # ( 1,  0)
        f7 = lambda x: 1./2.*(1 - x[0]**2)*(1 + x[1]*( 1)) # ( 0,  1)
        f8 = lambda x: 1./2.*(1 - x[1]**2)*(1 + x[0]*(-1)) # (-1,  0)
        shape_fun = [f1, f2, f3, f4, f5, f6, f7, f8]

        
    elif elem_type == 'CPE3' and dim == 2:
        f1 = lambda x: x[0]
        f2 = lambda x: x[1]
        f3 = lambda x: 1 - x[0] - x[1]
        shape_fun = [f1, f2, f3]
    
    elif elem_type == 'CPE6' and dim == 2:
        f1 = lambda x: x[0]*(2*x[0]-1)
        f2 = lambda x: x[1]*(2*x[1]-1)
        f3 = lambda x: (1-x[0]-x[1])*(1-2*x[0]-2*x[1])
        f4 = lambda x: 4*x[0]*x[1]
        f5 = lambda x: 4*x[1]*(1-x[0]-x[1])
        f6 = lambda x: 4*x[0]*(1-x[0]-x[1])
        shape_fun = [f1, f2, f3, f4, f5, f6]
    
    elif (elem_type == 'CPE4' or elem_type == 'CPE3') and dim == 1:
        f1 = lambda x: 0.5*(1-x[0])
        f2 = lambda x: 0.5*(1+x[0])
        shape_fun = [f1, f2]
        
    elif (elem_type == 'CPE8' or elem_type == 'CPE6') and dim == 1:
        f1 = lambda x: 0.5*x[0]*(x[0]-1)
        f2 = lambda x: -1.0*(x[0]+1)*(x[0]-1)
        f3 = lambda x: 0.5*x[0]*(x[0]+1)
        shape_fun = [f1, f2, f3]
    
    return shape_fun

def get_shape_grad_functions(elem_type, dim):
    shape_fns = get_shape_val_functions(elem_type, dim)
    return [jax.grad(f) for f in shape_fns]

@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type'])
def get_shape_vals(Gauss_Num, dim, elem_type):
    """Pre-compute shape function values

    Returns
    -------
    shape_vals: ndarray
        (8, 8) = (num_quads, num_nodes)  
    """
    shape_val_fns = get_shape_val_functions(elem_type, dim)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim, elem_type)
    shape_vals = []
    for quad_point in quad_points:
        physical_shape_vals = []
        for shape_val_fn in shape_val_fns:
            physical_shape_val = shape_val_fn(quad_point) 
            physical_shape_vals.append(physical_shape_val)
 
        shape_vals.append(physical_shape_vals)

    shape_vals = np.array(shape_vals) # (num_quads, num_nodes)
    # assert shape_vals.shape == (global_args['num_quads'], global_args['num_nodes'])
    return shape_vals

@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type'])
def get_shape_grads(Gauss_Num, dim, elem_type, XY, Elem_nodes):
    """Pre-compute shape function gradients

    Returns
    -------
    shape_grads_physical: ndarray
        (cell, num_quads, num_nodes, dim)  
    JxW: ndarray
        (cell, num_quads)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type, dim)
    quad_points, quad_weights = get_quad_points(Gauss_Num, dim, elem_type)
    # print('quad points weights')
    # print(quad_points, quad_weights)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            # See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
            # Page 147, Eq. (3.9.3)
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
 
        shape_grads.append(physical_shape_grads)
    
    
    shape_grads = np.array(shape_grads) # (num_quads, num_nodes, dim)
    # if dim == 2:
    #     print(quad_points)
    #     print(quad_weights)
    # if dim == 1:
    #     print(shape_grads.shape)
    #     print(shape_grads)
    # assert shape_grads.shape == (global_args['num_quads'], global_args['num_nodes'], global_args['dim'])
    
    physical_coos = np.take(XY, Elem_nodes, axis=0) # (num_cells, num_nodes, dim)
    # print('/n')
    # print(physical_coos.shape)
    # if dim == 1:
    #     print(physical_coos)
    # physical_coos: (num_cells, none,      num_nodes, dim, none)
    # shape_grads:   (none,      num_quads, num_nodes, none, dim)
    # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
    jacobian_dx_deta = np.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True)
    # if dim == 1:
    #     print(jacobian_dx_deta)
    
    jacobian_det = np.linalg.det(jacobian_dx_deta).reshape(len(Elem_nodes), len(quad_weights))# (num_cells, num_quads)
    # if dim == 2:
    #     print('jacob det')
    #     print(jacobian_det)
    
    jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
    # print(jacobian_deta_dx[0, :, 0, :, :])
    # print(shape_grads)
    shape_grads_physical = (shape_grads[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    # print(shape_grads_physical[0])

    # For first order FEM with 8 quad points, those quad weights are all equal to one
    # quad_weights = 1.
    c = 1
    if (elem_type == 'CPE3' or elem_type == 'CPE6') and dim == 2:
        c = 1/2
    JxW = jacobian_det * quad_weights[None, :] * c
    return shape_grads_physical, JxW # (num_cells, num_quads, num_nodes, dim), (num_cells, num_quads)

# compute FEM basic stuff
def get_A_b_FEM(XY, Elem_nodes, connectivity, nelem, Gauss_Num_FEM, quad_num_FEM, dim, elem_type, elem_dof, dof_global, 
                Cmat, Elem_nodes_tr, nelem_tr, Gauss_Num_tr, connectivity_tr, iffix):

    # decide how many blocks are we gonnna use
    size_BTB = nelem * quad_num_FEM * elem_dof * elem_dof
    nblock = int(size_BTB // max_array_size_block + 1)
    nelem_per_block_regular = nelem // nblock
    if nelem % nblock == 0:
        nelem_per_block_remainder = nelem_per_block_regular
    else:
        nelem_per_block_remainder = nelem_per_block_regular + nelem % nblock
    print(f"BTB array -> {nblock} blocks")
    # print(nelem, nblock, nelem_per_block_regular, nelem_per_block_remainder)
    
    # compute A_sp
    for iblock in range(nblock):
        if iblock == nblock-1:
            nelem_per_block = nelem_per_block_remainder
            elem_idx_block = np.array(range(nelem_per_block_regular*iblock, nelem_per_block_regular*iblock 
                                            + nelem_per_block_remainder), dtype=np.int32)
        else:
            nelem_per_block = nelem_per_block_regular
            elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
        Elem_nodes_block = Elem_nodes[elem_idx_block]
        # print('elem_nodes', nelem, nelem_per_block_regular, Elem_nodes_block)
        connectivity_block = connectivity[elem_idx_block]
        shape_grads_physical_block, JxW_block = get_shape_grads(Gauss_Num_FEM, dim, elem_type, XY, Elem_nodes_block)    
        
        Bmat_block = np.zeros((nelem_per_block, quad_num_FEM, 3, elem_dof), dtype=np.double)
        # print(JxW_block)
        # print(Bmat_block.shape)
        # print(shape_grads_physical_block.shape)
        # print(shape_grads_physical_block[0,0,:,:])
        if elem_type == 'CPE4':
            Bmat_block = Bmat_block.at[:,:,0,[0,2,4,6]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,1,[1,3,5,7]].set(shape_grads_physical_block[:,:,:,1])
            Bmat_block = Bmat_block.at[:,:,2,[1,3,5,7]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,2,[0,2,4,6]].set(shape_grads_physical_block[:,:,:,1])
        elif elem_type == 'CPE3':
            Bmat_block = Bmat_block.at[:,:,0,[0,2,4]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,1,[1,3,5]].set(shape_grads_physical_block[:,:,:,1])
            Bmat_block = Bmat_block.at[:,:,2,[1,3,5]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,2,[0,2,4]].set(shape_grads_physical_block[:,:,:,1])
        elif elem_type == 'CPE8':
            Bmat_block = Bmat_block.at[:,:,0,[0,2,4,6,8,10,12,14]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,1,[1,3,5,7,9,11,13,15]].set(shape_grads_physical_block[:,:,:,1])
            Bmat_block = Bmat_block.at[:,:,2,[1,3,5,7,9,11,13,15]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,2,[0,2,4,6,8,10,12,14]].set(shape_grads_physical_block[:,:,:,1])
        elif elem_type == 'CPE6':
            Bmat_block = Bmat_block.at[:,:,0,[0,2,4,6,8,10]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,1,[1,3,5,7,9,11]].set(shape_grads_physical_block[:,:,:,1])
            Bmat_block = Bmat_block.at[:,:,2,[1,3,5,7,9,11]].set(shape_grads_physical_block[:,:,:,0])
            Bmat_block = Bmat_block.at[:,:,2,[0,2,4,6,8,10]].set(shape_grads_physical_block[:,:,:,1])
        
        
        BTB_block = np.matmul(np.transpose(Bmat_block, (0,1,3,2)), Cmat[None,None,:,:]) # (num_cells, num_quads, num_nodes, num_nodes)
        BTB_block = np.matmul(BTB_block, Bmat_block)
        # print(BTB_block[0,0,:,:])
        # print(Cmat)
        # print(Bmat_block[0,0,:,:])
        V_block = np.sum(BTB_block * JxW_block[:, :, None, None], axis=1).reshape(-1) # (num_cells, num_nodes, num_nodes) -> (1 ,)
        I_block = np.repeat(connectivity_block, elem_dof, axis=1).reshape(-1)
        J_block = np.repeat(connectivity_block, elem_dof, axis=0).reshape(-1)
        
        if iblock == 0:
            A_sp_scipy =  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
        else:
            A_sp_scipy +=  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
    
    
    # Compute rhs or b, for traction force
    shape_vals_tr = get_shape_vals(Gauss_Num_tr, 1, elem_type) # dim == 1, linear element ()
    shape_grads_physical_tr, JxW_tr = get_shape_grads(Gauss_Num_tr, 1, elem_type, XY[:,[1]], Elem_nodes_tr)
    
    if elem_type == 'CPE4' or elem_type == 'CPE3':
        N_trac = np.zeros((nelem_tr, Gauss_Num_tr, 4, dim), dtype=np.double) # (nelem_tr, quad_num, elem_dof=2, dim)
        N_trac = N_trac.at[:,:,[0,2],0].set(shape_vals_tr[None,:,:]) # column 0
        N_trac = N_trac.at[:,:,[1,3],1].set(shape_vals_tr[None,:,:]) # column 1
    elif elem_type == 'CPE8' or elem_type == 'CPE6':
        N_trac = np.zeros((nelem_tr, Gauss_Num_tr, 6, dim), dtype=np.double) # (nelem_tr, quad_num, elem_dof=2, dim)
        N_trac = N_trac.at[:,:,[0,2,4],0].set(shape_vals_tr[None,:,:]) # column 0
        N_trac = N_trac.at[:,:,[1,3,5],1].set(shape_vals_tr[None,:,:]) # column 1
    
    NP = np.sum( np.matmul(N_trac, traction_force[None,None,:,:]) * JxW_tr[:,:,None,None], axis=1) # (nelem_tr, elem_dof, 1)
    rhs = np.zeros(dof_global, dtype=np.double)
    rhs = rhs.at[connectivity_tr.reshape(-1)].add(NP.reshape(-1))  # assemble 
    # print(rhs)
    
    # Apply boundary condition - Penalty method
    for idx, fix in enumerate(iffix):
        if fix:
            A_sp_scipy[idx,idx] *= 1e20
            rhs.at[idx].set(0)
    print(A_sp_scipy)
    # print(rhs)
    return A_sp_scipy, rhs


#%% ############### Convolutional FEM ###############


def get_adj_mat(Elem_nodes, nnode, s_patch):
    # Sparse matrix multiplication for graph theory.
    
    # get adjacency matrix of graph theory based on nodal connectivity
    # print(Elem_nodes)
    adj_rows, adj_cols = [], []
    # self 
    for inode in range(nnode):
        adj_rows += [inode]
        adj_cols += [inode]
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        for (inode, jnode) in combinations(list(elem_nodes), 2):
            adj_rows += [inode, jnode]
            adj_cols += [jnode, inode]
    adj_values = onp.ones(len(adj_rows), dtype=onp.int32)
    adj_rows = onp.array(adj_rows, dtype=onp.int32)
    adj_cols = onp.array(adj_cols, dtype=onp.int32)
    # print(nnode)
    # print(adj_values.shape)
    # print(adj_rows)
    # print(adj_cols.shape)
    
    # build sparse matrix
    adj_sp = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    adj_s = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (nnode, nnode))
    # print(adj_sp.toarray())
    
    # compute s th power of the adjacency matrix to get s th order of connectivity
    for itr in range(s_patch-1):
        adj_s = adj_s.dot(adj_sp)
    indices = adj_s.indices
    indptr = adj_s.indptr
    return indices, indptr

# def nodal_patch(jnode, elemental_patch_nodes):
    # return onp.where(elemental_patch_nodes==jnode)[0]
    

def get_dex_max(indices, indptr, s_patch, d_c, XY, Elem_nodes, nelem, nnode, nodes_per_elem):
    edex_max = (2+2*s_patch)**2 # estimated value of edex_max
    Elemental_patch_nodes_st = (-1)*onp.ones((nelem, edex_max+20), dtype=onp.int32) # giving extra +20 spaces 
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        if len(elem_nodes) == 4:
            elemental_patch_nodes = onp.unique(onp.concatenate((indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ],  # node_idx 0
                                                                indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ],  # node_idx 1
                                                                indices[ indptr[elem_nodes[2]] : indptr[elem_nodes[2]+1] ],  # node_idx 2
                                                                indices[ indptr[elem_nodes[3]] : indptr[elem_nodes[3]+1] ])))  # node_idx 3
        elif len(elem_nodes) == 3:
            elemental_patch_nodes = onp.unique(onp.concatenate((indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ],  # node_idx 0
                                                                indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ],  # node_idx 1
                                                                indices[ indptr[elem_nodes[2]] : indptr[elem_nodes[2]+1] ])))  # node_idx 2
        edex = len(elemental_patch_nodes)
        edexes[ielem] = edex
        Elemental_patch_nodes_st[ielem, :edex] = elemental_patch_nodes
    
    edex_max = onp.max(edexes)
    Elemental_patch_nodes_st = Elemental_patch_nodes_st[:,:edex_max]  # static, (num_elements, edex_max)
    
    # computes nodal support domain info and vmap_inputs
    # part1_time, part2_time = 0, 0
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (num_cells, num_nodes)
    
    dist_patch = (s_patch+0.9)*d_c
    for ielem, (elem_nodes, edex, elemental_patch_nodes_st) in enumerate(zip(
            Elem_nodes, edexes, Elemental_patch_nodes_st)):
        
        elemental_patch_nodes = elemental_patch_nodes_st[:edex]
        
        for inode_idx, inode in enumerate(elem_nodes):
            
            dist_mat = onp.absolute(XY[elemental_patch_nodes,:] - XY[inode,:])
            nodal_patch_nodes_idx = onp.where((dist_mat[:,0]<=dist_patch) & (dist_mat[:,1]<=dist_patch))[0]
            # nodal_patch_nodes = elemental_patch_nodes[nodal_patch_nodes_idx]
            ndex = len(nodal_patch_nodes_idx)
            ndexes[ielem, inode_idx] = ndex
            
    ndex_max = onp.max(ndexes)
    return edex_max, ndex_max

def get_patch_info(indices, indptr, s_patch, d_c, edex_max, ndex_max, XY, Elem_nodes, nelem, nodes_per_elem):
    
    # Elemental patch, ~s
    Elemental_patch_nodes_st = onp.zeros((nelem, edex_max), dtype=onp.int32) # edex_max should be grater than 100!
    edexes = onp.zeros(nelem, dtype=onp.int32) # (num_elements, )
    
    for ielem, elem_nodes in enumerate(Elem_nodes):
        if len(elem_nodes) == 4:
            elemental_patch_nodes = onp.unique(onp.concatenate((indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ],  # node_idx 0
                                                                indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ],  # node_idx 1
                                                                indices[ indptr[elem_nodes[2]] : indptr[elem_nodes[2]+1] ],  # node_idx 2
                                                                indices[ indptr[elem_nodes[3]] : indptr[elem_nodes[3]+1] ])))  # node_idx 3
        elif len(elem_nodes) == 3:
            elemental_patch_nodes = onp.unique(onp.concatenate((indices[ indptr[elem_nodes[0]] : indptr[elem_nodes[0]+1] ],  # node_idx 0
                                                                indices[ indptr[elem_nodes[1]] : indptr[elem_nodes[1]+1] ],  # node_idx 1
                                                                indices[ indptr[elem_nodes[2]] : indptr[elem_nodes[2]+1] ])))  # node_idx 2
        edex = len(elemental_patch_nodes)
        edexes[ielem] = edex
        Elemental_patch_nodes_st[ielem, :edex] = elemental_patch_nodes
    
    Nodal_patch_nodes_st = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (num_cells, num_nodes, ndex_max)
    Nodal_patch_nodes_bool = onp.zeros((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (num_cells, num_nodes, ndex_max)
    Nodal_patch_nodes_idx = (-1)*onp.ones((nelem, nodes_per_elem, ndex_max), dtype=onp.int32) # static, (num_cells, num_nodes, ndex_max)
    ndexes = onp.zeros((nelem, nodes_per_elem), dtype=onp.int32) # static, (num_cells, num_nodes)
    
    # Nodal patch, ~a
    dist_patch = (s_patch+0.9)*d_c
    for ielem, (elem_nodes, edex, elemental_patch_nodes_st) in enumerate(zip(
            Elem_nodes, edexes, Elemental_patch_nodes_st)):
        
        elemental_patch_nodes = elemental_patch_nodes_st[:edex]
        
        for inode_idx, inode in enumerate(elem_nodes):
            
            dist_mat = onp.absolute(XY[elemental_patch_nodes,:] - XY[inode,:])
            nodal_patch_nodes_idx = onp.where((dist_mat[:,0]<=dist_patch) & (dist_mat[:,1]<=dist_patch))[0]
            nodal_patch_nodes = elemental_patch_nodes[nodal_patch_nodes_idx]
            ndex = len(nodal_patch_nodes)
            
            Nodal_patch_nodes_st[ielem, inode_idx, :ndex] = nodal_patch_nodes  # convert to global nodes
            Nodal_patch_nodes_bool[ielem, inode_idx, :ndex] = onp.where(nodal_patch_nodes>=0, 1, 0)
            Nodal_patch_nodes_idx[ielem, inode_idx, :ndex] = nodal_patch_nodes_idx
            ndexes[ielem, inode_idx] = ndex
            
    # Convert everything to device array
    Elemental_patch_nodes_st = np.array(Elemental_patch_nodes_st)
    edexes = np.array(edexes)
    Nodal_patch_nodes_st = np.array(Nodal_patch_nodes_st)
    Nodal_patch_nodes_bool = np.array(Nodal_patch_nodes_bool)
    Nodal_patch_nodes_idx = np.array(Nodal_patch_nodes_idx)
    ndexes = np.array(ndexes)
    
    return Elemental_patch_nodes_st, edexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes

@partial(jax.jit, static_argnames=['ndex_max','mbasis']) # This will slower the function
def Compute_RadialBasis_1D(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                         a_dil, mbasis=0):
    # x, y: point of interest
    # xv: [nx - dim X ndex] support domain nodal coords.
    # ndex: number of supporting nodes
    # R: Shape parameter, R = alpha_c * d_c
    # q: shape parameter
    # nRBF: type of RBF, 1-MQ, 2-Cubic spline
    # a_dil: dilation parameter for cubic spline
    # mbasis: number of polynomial terms
    # derivative: order of derivative
    
    RP = np.zeros(ndex_max + mbasis, dtype=np.double)
    

    for i in range(ndex_max):
        ndex_bool = nodal_patch_nodes_bool[i]
        zI = np.linalg.norm(xy - xv[i])/a_dil # by defining support domain, zI is bounded btw 0 and 1.
        
        bool1 = np.ceil(0.50001-zI) # returns 1 when 0 < zI <= 0.5 and 0 when 0.5 < zI <1
        bool2 = np.ceil(zI-0.50001) # returns 0 when 0 < zI <= 0.5 and 1 when 0.5 < zI <1
        bool3 = np.heaviside(1-zI, 1) # returns 1 when zI <= 1 // 0 when 1 < zI
                
        # Cubic spline
        RP = RP.at[i].add( ((2/3 - 4*zI**2 + 4*zI**3         ) * bool1 +    # phi_i
                            (4/3 - 4*zI + 4*zI**2 - 4/3*zI**3) * bool2) * bool3 * ndex_bool)   # phi_i
        
        # # Wendland C2
        # RP = RP.at[i].add( (1-zI)**4*(1+4*zI) * bool3 * ndex_bool)   # phi_i
        
        
    if mbasis > 0: # 1st
        RP = RP.at[ndex].set(1)   # N
        RP = RP.at[ndex+1].set(xy[0])   # N
        RP = RP.at[ndex+2].set(xy[1])   # N
            
    if mbasis > 3: # 2nd
        RP = RP.at[ndex+3].set(xy[0]    * xy[1]   )  # N
        RP = RP.at[ndex+4].set(xy[0]**2           )  # N
        RP = RP.at[ndex+5].set(           xy[1]**2)  # N
            
    if mbasis > 6: # 3rd
        RP = RP.at[ndex+6].set(xy[0]**2 * xy[1]   )     # N
        RP = RP.at[ndex+7].set(xy[0]    * xy[1]**2)     # N
        RP = RP.at[ndex+8].set(xy[0]**3           )          # N
        RP = RP.at[ndex+9].set(           xy[1]**3)          # N
            
    if mbasis > 10: # 4th
        RP = RP.at[ndex+10].set(xy[0]**2 * xy[1]**2)     # N
        RP = RP.at[ndex+11].set(xy[0]**3 * xy[1]   )     # N
        RP = RP.at[ndex+12].set(xy[0]    * xy[1]**3)     # N 
        RP = RP.at[ndex+13].set(xy[0]**4           )     # N
        RP = RP.at[ndex+14].set(           xy[1]**4)     # N

    if mbasis > 15: # 5th
        RP = RP.at[ndex+15].set(xy[0]**3 * xy[1]**2)     # N
        RP = RP.at[ndex+16].set(xy[0]**2 * xy[1]**3)     # N
        RP = RP.at[ndex+17].set(xy[0]**4 * xy[1]   )     # N 
        RP = RP.at[ndex+18].set(xy[0]    * xy[1]**4)     # N 
        RP = RP.at[ndex+19].set(xy[0]**5           )     # N
        RP = RP.at[ndex+20].set(           xy[1]**5)     # N

    if mbasis > 21: # 6th
        RP = RP.at[ndex+21].set(xy[0]**3 * xy[1]**3)     # N
        RP = RP.at[ndex+22].set(xy[0]**4 * xy[1]**2)     # N
        RP = RP.at[ndex+23].set(xy[0]**2 * xy[1]**4)     # N 
        RP = RP.at[ndex+24].set(xy[0]**5 * xy[1]   )     # N 
        RP = RP.at[ndex+25].set(xy[0]    * xy[1]**5)     # N 
        RP = RP.at[ndex+26].set(xy[0]**6           )     # N
        RP = RP.at[ndex+27].set(           xy[1]**6)     # N

    if mbasis > 28: # 7th
        RP = RP.at[ndex+28].set(xy[0]**4 * xy[1]**3)     # N
        RP = RP.at[ndex+29].set(xy[0]**3 * xy[1]**4)     # N
        RP = RP.at[ndex+30].set(xy[0]**5 * xy[1]**2)     # N 
        RP = RP.at[ndex+31].set(xy[0]**2 * xy[1]**5)     # N 
        RP = RP.at[ndex+32].set(xy[0]**6 * xy[1]   )     # N 
        RP = RP.at[ndex+33].set(xy[0]    * xy[1]**6)     # N 
        RP = RP.at[ndex+34].set(xy[0]**7           )     # N
        RP = RP.at[ndex+35].set(           xy[1]**7)     # N

    if mbasis > 36: # 8th
        RP = RP.at[ndex+36].set(xy[0]**4 * xy[1]**4)     # N
        RP = RP.at[ndex+37].set(xy[0]**5 * xy[1]**3)     # N
        RP = RP.at[ndex+38].set(xy[0]**3 * xy[1]**5)     # N 
        RP = RP.at[ndex+39].set(xy[0]**6 * xy[1]**2)     # N 
        RP = RP.at[ndex+40].set(xy[0]**2 * xy[1]**6)     # N 
        RP = RP.at[ndex+41].set(xy[0]**7 * xy[1]   )     # N 
        RP = RP.at[ndex+42].set(xy[0]    * xy[1]**7)     # N 
        RP = RP.at[ndex+43].set(xy[0]**8           )     # N
        RP = RP.at[ndex+44].set(           xy[1]**8)     # N

            
    return RP

@partial(jax.jit, static_argnames=['ndex_max','mbasis']) # This will slower the function
def Compute_RadialBasis(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                         a_dil, mbasis=0):
    # x, y: point of interest
    # xv: [nx - dim X ndex] support domain nodal coords.
    # ndex: number of supporting nodes
    # R: Shape parameter, R = alpha_c * d_c
    # q: shape parameter
    # nRBF: type of RBF, 1-MQ, 2-Cubic spline
    # a_dil: dilation parameter for cubic spline
    # mbasis: number of polynomial terms
    # derivative: order of derivative
    
    # if derivative == 0:
        # RP = np.zeros((ndex_max + mbasis, 1), dtype=np.double)
    # elif derivative == 1:
    RP = np.zeros((ndex_max + mbasis, 3), dtype=np.double)


    for i in range(ndex_max):
        ndex_bool = nodal_patch_nodes_bool[i]
        zI = np.linalg.norm(xy - xv[i])/a_dil # by defining support domain, zI is bounded btw 0 and 1.
        
        bool1 = np.ceil(0.50001-zI) # returns 1 when 0 < zI <= 0.5 and 0 when 0.5 < zI <1
        bool2 = np.ceil(zI-0.50001) # returns 0 when 0 < zI <= 0.5 and 1 when 0.5 < zI <1
        bool3 = np.heaviside(1-zI, 1) # returns 1 when zI <= 1 // 0 when 1 < zI
        
        # Cubic spline
        RP = RP.at[i,0].add( ((2/3 - 4*zI**2 + 4*zI**3         ) * bool1 +    # phi_i
                              (4/3 - 4*zI + 4*zI**2 - 4/3*zI**3) * bool2) * bool3 * ndex_bool)   # phi_i
        dzIdx = (xy[0] - xv[i,0])/(a_dil**2*zI)
        dzIdy = (xy[1] - xv[i,1])/(a_dil**2*zI)
        RP = RP.at[i,1].add( ((-8*zI + 12*zI**2   )*dzIdx * bool1 +  # phi_i,x
                              (-4 + 8*zI - 4*zI**2)*dzIdx * bool2) * bool3 * ndex_bool) # phi_i,x
        RP = RP.at[i,2].add( ((-8*zI + 12*zI**2   )*dzIdy * bool1 + # phi_i,y
                              (-4 + 8*zI - 4*zI**2)*dzIdy * bool2) * bool3 * ndex_bool) # phi_i,y
                
        # # Wendland C2
        # RP = RP.at[i].add( (1-zI)**4*(1+4*zI) * bool3 * ndex_bool)   # phi_i
        # dzIdx = (xy[0] - xv[i,0])/(a_dil**2*zI)
        # dzIdy = (xy[1] - xv[i,1])/(a_dil**2*zI)
        # RP = RP.at[i,1].add( -20*zI*(1-zI)**3*dzIdx * bool3 * ndex_bool) # phi_i,x
        # RP = RP.at[i,2].add( -20*zI*(1-zI)**3*dzIdy * bool3 * ndex_bool) # phi_i,y
        
    
    if mbasis > 0: # 1st
        RP = RP.at[ndex, 0].set(1)   # N
        RP = RP.at[ndex+1, 0].set(xy[0])   # N
        RP = RP.at[ndex+2, 0].set(xy[1])   # N
        # if derivative > 0:
        RP = RP.at[ndex+1, 1].set(1)   # dNdx
        RP = RP.at[ndex+2, 2].set(1)   # dNdy
            
    if mbasis > 3: # 2nd
        RP = RP.at[ndex+3, 0].set(xy[0]    * xy[1]   )  # N
        RP = RP.at[ndex+4, 0].set(xy[0]**2           )  # N
        RP = RP.at[ndex+5, 0].set(           xy[1]**2)  # N
        # if derivative > 0:
        RP = RP.at[ndex+3, 1].set(xy[1])       # dNdx
        RP = RP.at[ndex+4, 1].set(2*xy[0])     # dNdx
        RP = RP.at[ndex+3, 2].set(xy[0])       # dNdy
        RP = RP.at[ndex+5, 2].set(2*xy[1])     # dNdy
            
    if mbasis > 6: # 3rd
        RP = RP.at[ndex+6, 0].set(xy[0]**2 * xy[1]   )     # N
        RP = RP.at[ndex+7, 0].set(xy[0]    * xy[1]**2)     # N
        RP = RP.at[ndex+8, 0].set(xy[0]**3           )          # N
        RP = RP.at[ndex+9, 0].set(           xy[1]**3)          # N
        # if derivative > 0:
        RP = RP.at[ndex+6, 1].set(2*xy[0]    * xy[1]   )    # dNdx
        RP = RP.at[ndex+7, 1].set(             xy[1]**2)    # dNdx
        RP = RP.at[ndex+8, 1].set(3*xy[0]**2           )    # dNdx
        RP = RP.at[ndex+6, 2].set(xy[0]**2             )    # dNdy
        RP = RP.at[ndex+7, 2].set(2*xy[0]    * xy[1]   )    # dNdy
        RP = RP.at[ndex+9, 2].set(3          * xy[1]**2)    # dNdy
            
    if mbasis > 10: # 4th
        RP = RP.at[ndex+10, 0].set(xy[0]**2 * xy[1]**2)     # N
        RP = RP.at[ndex+11, 0].set(xy[0]**3 * xy[1]   )     # N
        RP = RP.at[ndex+12, 0].set(xy[0]    * xy[1]**3)     # N 
        RP = RP.at[ndex+13, 0].set(xy[0]**4           )     # N
        RP = RP.at[ndex+14, 0].set(           xy[1]**4)     # N
        # if derivative > 0:
        RP = RP.at[ndex+10, 1].set(2*xy[0]    * xy[1]**2)       # dNdx
        RP = RP.at[ndex+11, 1].set(3*xy[0]**2 * xy[1]   )       # dNdx
        RP = RP.at[ndex+12, 1].set(             xy[1]**3)       # dNdx
        RP = RP.at[ndex+13, 1].set(4*xy[0]**3           )       # dNdx
        RP = RP.at[ndex+10, 2].set(2*xy[0]**2 * xy[1]   )       # dNdy
        RP = RP.at[ndex+11, 2].set(  xy[0]**3           )       # dNdy
        RP = RP.at[ndex+12, 2].set(3*xy[0]    * xy[1]**2)       # dNdy
        RP = RP.at[ndex+14, 2].set(4*           xy[1]**3)       # dNdy
    if mbasis > 15: # 5th
        RP = RP.at[ndex+15, 0].set(xy[0]**3 * xy[1]**2)     # N
        RP = RP.at[ndex+16, 0].set(xy[0]**2 * xy[1]**3)     # N
        RP = RP.at[ndex+17, 0].set(xy[0]**4 * xy[1]   )     # N 
        RP = RP.at[ndex+18, 0].set(xy[0]    * xy[1]**4)     # N 
        RP = RP.at[ndex+19, 0].set(xy[0]**5           )     # N
        RP = RP.at[ndex+20, 0].set(           xy[1]**5)     # N
        # if derivative > 0:
        RP = RP.at[ndex+15, 1].set(3*xy[0]**2 * xy[1]**2)       # dNdx
        RP = RP.at[ndex+16, 1].set(2*xy[0]    * xy[1]**3)       # dNdx
        RP = RP.at[ndex+17, 1].set(4*xy[0]**3 * xy[1]   )       # dNdx
        RP = RP.at[ndex+18, 1].set(             xy[1]**4)       # dNdx
        RP = RP.at[ndex+19, 1].set(5*xy[0]**4           )       # dNdx
        
        RP = RP.at[ndex+15, 2].set(2*xy[0]**3 * xy[1]   )       # dNdy
        RP = RP.at[ndex+16, 2].set(3*xy[0]**2 * xy[1]**2)       # dNdy
        RP = RP.at[ndex+17, 2].set(  xy[0]**4           )       # dNdy
        RP = RP.at[ndex+18, 2].set(4*xy[0]    * xy[1]**3)       # dNdy
        RP = RP.at[ndex+20, 2].set(5*           xy[1]**4)       # dNdy
    if mbasis > 21: # 6th
        RP = RP.at[ndex+21, 0].set(xy[0]**3 * xy[1]**3)     # N
        RP = RP.at[ndex+22, 0].set(xy[0]**4 * xy[1]**2)     # N
        RP = RP.at[ndex+23, 0].set(xy[0]**2 * xy[1]**4)     # N 
        RP = RP.at[ndex+24, 0].set(xy[0]**5 * xy[1]   )     # N 
        RP = RP.at[ndex+25, 0].set(xy[0]    * xy[1]**5)     # N 
        RP = RP.at[ndex+26, 0].set(xy[0]**6           )     # N
        RP = RP.at[ndex+27, 0].set(           xy[1]**6)     # N
        # if derivative > 0:
        RP = RP.at[ndex+21, 1].set(3*xy[0]**2 * xy[1]**3)       # dNdx
        RP = RP.at[ndex+22, 1].set(4*xy[0]**3 * xy[1]**2)       # dNdx
        RP = RP.at[ndex+23, 1].set(2*xy[0]    * xy[1]**4)       # dNdx
        RP = RP.at[ndex+24, 1].set(5*xy[0]**4 * xy[1]   )       # dNdx
        RP = RP.at[ndex+25, 1].set(             xy[1]**5)       # dNdx
        RP = RP.at[ndex+26, 1].set(6*xy[0]**5           )       # dNdx
        
        RP = RP.at[ndex+21, 2].set(3*xy[0]**3 * xy[1]**2)       # dNdy
        RP = RP.at[ndex+22, 2].set(2*xy[0]**4 * xy[1]   )       # dNdy
        RP = RP.at[ndex+23, 2].set(4*xy[0]**2 * xy[1]**3)       # dNdy
        RP = RP.at[ndex+24, 2].set(  xy[0]**5           )       # dNdy
        RP = RP.at[ndex+25, 2].set(5*xy[0]    * xy[1]**4)       # dNdy
        RP = RP.at[ndex+27, 2].set(6*           xy[1]**5)       # dNdy
    if mbasis > 28: # 7th
        RP = RP.at[ndex+28, 0].set(xy[0]**4 * xy[1]**3)     # N
        RP = RP.at[ndex+29, 0].set(xy[0]**3 * xy[1]**4)     # N
        RP = RP.at[ndex+30, 0].set(xy[0]**5 * xy[1]**2)     # N 
        RP = RP.at[ndex+31, 0].set(xy[0]**2 * xy[1]**5)     # N 
        RP = RP.at[ndex+32, 0].set(xy[0]**6 * xy[1]   )     # N 
        RP = RP.at[ndex+33, 0].set(xy[0]    * xy[1]**6)     # N 
        RP = RP.at[ndex+34, 0].set(xy[0]**7           )     # N
        RP = RP.at[ndex+35, 0].set(           xy[1]**7)     # N
        # if derivative > 0:            
        RP = RP.at[ndex+28, 1].set(4*xy[0]**3 * xy[1]**3)       # dNdx
        RP = RP.at[ndex+29, 1].set(3*xy[0]**2 * xy[1]**4)       # dNdx
        RP = RP.at[ndex+30, 1].set(5*xy[0]**4 * xy[1]**2)       # dNdx
        RP = RP.at[ndex+31, 1].set(2*xy[0]    * xy[1]**5)       # dNdx
        RP = RP.at[ndex+32, 1].set(6*xy[0]**5 * xy[1]   )       # dNdx
        RP = RP.at[ndex+33, 1].set(             xy[1]**6)       # dNdx
        RP = RP.at[ndex+34, 1].set(7*xy[0]**6           )       # dNdx
        
        RP = RP.at[ndex+28, 2].set(3*xy[0]**4 * xy[1]**2)       # dNdy
        RP = RP.at[ndex+29, 2].set(4*xy[0]**3 * xy[1]**3)       # dNdy
        RP = RP.at[ndex+30, 2].set(2*xy[0]**5 * xy[1]   )       # dNdy
        RP = RP.at[ndex+31, 2].set(5*xy[0]**2 * xy[1]**4)       # dNdy
        RP = RP.at[ndex+32, 2].set(  xy[0]**6           )       # dNdy
        RP = RP.at[ndex+33, 2].set(6*xy[0]    * xy[1]**5)       # dNdy
        RP = RP.at[ndex+35, 2].set(7*           xy[1]**6)       # dNdy
    if mbasis > 36: # 8th
        RP = RP.at[ndex+36, 0].set(xy[0]**4 * xy[1]**4)     # N
        RP = RP.at[ndex+37, 0].set(xy[0]**5 * xy[1]**3)     # N
        RP = RP.at[ndex+38, 0].set(xy[0]**3 * xy[1]**5)     # N 
        RP = RP.at[ndex+39, 0].set(xy[0]**6 * xy[1]**2)     # N 
        RP = RP.at[ndex+40, 0].set(xy[0]**2 * xy[1]**6)     # N 
        RP = RP.at[ndex+41, 0].set(xy[0]**7 * xy[1]   )     # N 
        RP = RP.at[ndex+42, 0].set(xy[0]    * xy[1]**7)     # N 
        RP = RP.at[ndex+43, 0].set(xy[0]**8           )     # N
        RP = RP.at[ndex+44, 0].set(           xy[1]**8)     # N
        # if derivative > 0:            
        RP = RP.at[ndex+36, 1].set(4*xy[0]**3 * xy[1]**4)       # dNdx
        RP = RP.at[ndex+37, 1].set(5*xy[0]**4 * xy[1]**3)       # dNdx
        RP = RP.at[ndex+38, 1].set(3*xy[0]**2 * xy[1]**5)       # dNdx
        RP = RP.at[ndex+39, 1].set(6*xy[0]**5 * xy[1]**2)       # dNdx
        RP = RP.at[ndex+40, 1].set(2*xy[0]    * xy[1]**6)       # dNdx
        RP = RP.at[ndex+41, 1].set(7*xy[0]**6 * xy[1]   )       # dNdx
        RP = RP.at[ndex+42, 1].set(             xy[1]**7)       # dNdx
        RP = RP.at[ndex+43, 1].set(8*xy[0]**7           )       # dNdx
        
        RP = RP.at[ndex+36, 2].set(4*xy[0]**4 * xy[1]**3)       # dNdy
        RP = RP.at[ndex+37, 2].set(3*xy[0]**5 * xy[1]**2)       # dNdy
        RP = RP.at[ndex+38, 2].set(5*xy[0]**3 * xy[1]**4)       # dNdy
        RP = RP.at[ndex+39, 2].set(2*xy[0]**6 * xy[1]   )       # dNdy
        RP = RP.at[ndex+40, 2].set(6*xy[0]**2 * xy[1]**5)       # dNdy
        RP = RP.at[ndex+41, 2].set(  xy[0]**7           )       # dNdy
        RP = RP.at[ndex+42, 2].set(7*xy[0]    * xy[1]**6)       # dNdy
        RP = RP.at[ndex+44, 2].set(8*           xy[1]**7)       # dNdy

            
    return RP

# @partial(jax.jit, static_argnames=['ndex_max','mbasis'])
def get_G(xv, ndex, ndex_max, nodal_patch_nodes_bool, a_dil, mbasis):
    # nodal_patch_nodes_bool: (ndex_max,)
    G = np.zeros((ndex_max + mbasis, ndex_max + mbasis), dtype=np.double)

    # Build RP
    for idx, (X, ndex_bool) in enumerate(zip(xv, nodal_patch_nodes_bool)):

        RP = Compute_RadialBasis_1D(X, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                                  a_dil, mbasis) # (edex_max,) but only 'ndex+1' nonzero terms
        G = G.at[:,idx].set(RP * ndex_bool)             
    
    # Make symmetric matrix
    G = np.tril(G) + np.triu(G.T, 1)
    # Build diagonal terms to nullify dimensions
    for idx, ndex_bool in enumerate(nodal_patch_nodes_bool):
        G = G.at[idx + mbasis, idx + mbasis].add(abs(ndex_bool-1))
        
    return G # G matrix


def get_Gs(vmap_input_G, 
            Nodal_patch_nodes_st, Nodal_patch_nodes_bool, ndexes, ndex_max,
            XY,  a_dil, mbasis):
    
    ielem = vmap_input_G[0]
    inode_idx = vmap_input_G[1]
    
    ndex = ndexes[ielem, inode_idx]
    nodal_patch_nodes = Nodal_patch_nodes_st[ielem, inode_idx, :] # static
    nodal_patch_nodes_bool = Nodal_patch_nodes_bool[ielem, inode_idx, :] # static
    
    xv = XY[nodal_patch_nodes,:]
    G = get_G(xv, ndex, ndex_max, nodal_patch_nodes_bool,  a_dil, mbasis)
    
    return G

@partial(jax.jit, static_argnames=['edex_max','ndex_max','mbasis'])
def get_phi(vmap_input, elem_idx, Gs, shape_vals, edex_max, # 5
            Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes, ndex_max, # 5
            XY, Elem_nodes, a_dil, mbasis): # 7
    
    ielem_idx = vmap_input[0]
    ielem = elem_idx[ielem_idx]
    iquad = vmap_input[1]
    inode_idx = vmap_input[2]
    shape_val = shape_vals[iquad,:]

    elem_nodes = Elem_nodes[ielem, :]
    xy_elem = XY[elem_nodes,:] # (num_nodes, dim)
    
    ndex = ndexes[ielem_idx, inode_idx]
    nodal_patch_nodes = Nodal_patch_nodes_st[ielem_idx, inode_idx, :] # static
    nodal_patch_nodes_bool = Nodal_patch_nodes_bool[ielem_idx, inode_idx, :] # static
    nodal_patch_nodes_idx = Nodal_patch_nodes_idx[ielem_idx, inode_idx, :] # static

    xv = XY[nodal_patch_nodes,:]
    G = Gs[ielem_idx, inode_idx, :, :]
    
    xy = np.sum(shape_val[:, None] * xy_elem, axis=0, keepdims=False)
    RP = Compute_RadialBasis(xy, xv, ndex, ndex_max, nodal_patch_nodes_bool, 
                              a_dil, mbasis)
    phi_org = np.linalg.solve(G.T, RP)[:ndex_max,:] * nodal_patch_nodes_bool[:, None]
    phi = np.zeros((edex_max + 1, 1+dim))  # trick, add dummy node at the end
    phi = phi.at[nodal_patch_nodes_idx, :].set(phi_org) 
    phi = phi[:edex_max,:] # trick, delete dummy node
    
    return phi

def get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                       XY, XY_host, Elem_nodes, Elem_nodes_host, shape_vals, Gauss_Num_CFEM, quad_num_CFEM, dim, elem_type, nodes_per_elem,
                       indices, indptr, s_patch, d_c, edex_max, ndex_max,
                        a_dil, mbasis):
    
    # time_patch, time_G, time_Phi = 0,0,0
    elem_idx_block_host = onp.array(elem_idx_block) # cpu
    elem_idx_block = np.array(elem_idx_block) # gpu
    Elem_nodes_block_host = Elem_nodes_host[elem_idx_block_host] # cpu
    Elem_nodes_block = Elem_nodes[elem_idx_block] # gpu
    shape_grads_physical_block, JxW_block = get_shape_grads(Gauss_Num_CFEM, dim, elem_type, XY, Elem_nodes_block) # gpu
    # if dim == 2:
    #     print('at dim2')
    #     print(shape_grads_physical_block.shape)
    #     print(JxW_block.shape)
    
    # start_patch = time.time()
    (Elemental_patch_nodes_st_block, edexes_block,
     Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, 
     ndexes_block) = get_patch_info(indices, indptr, s_patch, d_c, edex_max, ndex_max, XY_host, Elem_nodes_block_host, 
                                    nelem_per_block, nodes_per_elem)                         
    # print(Nodal_patch_nodes_st_block)
    # print(Elemental_patch_nodes_st_block)
    # print(edexes_block)
    # time_patch += (time.time() - start_patch)
    
    #############  get G #############
    # start_G = time.time()
    # compute moment matrix for the meshfree shape functions
    vmap_inputs_G = np.concatenate((np.repeat(np.arange(nelem_per_block), nodes_per_elem), # ielem
                      np.tile(np.arange(nodes_per_elem), nelem_per_block))).reshape(2,-1).T # inode
    
    # print(vmap_inputs_G)
    Gs = jax.vmap(get_Gs, in_axes = (0, 
            None, None, None, None, 
            None, None, None), out_axes = 0
            )(vmap_inputs_G,
              Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, ndexes_block, ndex_max,
              XY, a_dil, mbasis) ########################################### XY_norm!!
    Gs = np.reshape(Gs, (nelem_per_block, nodes_per_elem, ndex_max+mbasis, ndex_max+mbasis))
    # Gs: (num_cells, num_nodes, ndex_max+mbasis, ndex_max+mbasis)
    # time_G += (time.time() - start_G)
    
    ############# Phi ###############
    # start_Phi = time.time()
    # compute meshfree shape functions
    # linearize vmap inputs (ielem, iquad, inode)
    vmap_inputs = np.concatenate((np.repeat(np.arange(nelem_per_block), quad_num_CFEM*nodes_per_elem), # ielem_idx
                      np.tile(np.repeat(np.arange(quad_num_CFEM), nodes_per_elem), nelem_per_block), # iquad
                      np.tile(np.arange(nodes_per_elem), nelem_per_block*quad_num_CFEM))).reshape(3,-1).T # inode
    
    # print(vmap_inputs)
    # Vmap for CFEM shape functions for regular elements
    Phi = jax.vmap(get_phi, in_axes = (0, None, None, None, None, 
            None, None, None, None, None, 
            None, None, None, None), out_axes = 0
            )(vmap_inputs, elem_idx_block, Gs, shape_vals, edex_max,
            Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, ndexes_block, ndex_max,
            XY, Elem_nodes, a_dil, mbasis) ########################################### XY_norm!!
    
    
    
    # (num_cells, num_quads, num_nodes, edex_max, 1+dim)
    Phi = np.reshape(Phi, (nelem_per_block, quad_num_CFEM, nodes_per_elem, edex_max, 1+dim))
    N_til_block = np.sum(shape_vals[None, :, :, None]*Phi[:,:,:,:,0], axis=2) # (num_cells, num_quads, edex_max)
    Grad_N_til_block = (np.sum(shape_grads_physical_block[:, :, :, None, :]*Phi[:,:,:,:,:1], axis=2) 
                      + np.sum(shape_vals[None, :, :, None, None]*Phi[:,:,:,:,1:], axis=2) )
    
    # Check partition of unity
    # if 0 in elem_idx_block: 
    if not ( np.allclose(np.sum(N_til_block, axis=2), np.ones((nelem_per_block, quad_num_CFEM), dtype=np.double)) and
            np.allclose(np.sum(Grad_N_til_block, axis=2), np.zeros((nelem_per_block, quad_num_CFEM, dim), dtype=np.double)) ):
        print(f"PoU Check failed at element {elem_idx_block[0]}~{elem_idx_block[-1]}")
        # print(np.sum(N_til_block, axis=2))
        PoU_Check_N = (np.linalg.norm(np.sum(N_til_block, axis=2) - 
                    np.ones((nelem_per_block, quad_num_CFEM), dtype=np.float64))**2/(nelem_per_block*quad_num_CFEM))**0.5
        PoU_Check_Grad_N = (np.linalg.norm(np.sum(Grad_N_til_block, axis=2))**2/(nelem_per_block*quad_num_CFEM*dim))**0.5
        print(f'PoU check N / Grad_N: {PoU_Check_N:.4e} / {PoU_Check_Grad_N:.4e}')
        
        
    return N_til_block, Grad_N_til_block, JxW_block, Elemental_patch_nodes_st_block, Elem_nodes_block


def get_connectivity(Elemental_patch_nodes, dim):
    # get connectivity vector for 2D element
    (nelem, edex_max) = Elemental_patch_nodes.shape
    connectivity = np.zeros((nelem, edex_max*dim), dtype = np.int64)
    connectivity = connectivity.at[:, np.arange(0,edex_max*dim, 2)].set(Elemental_patch_nodes*2)
    connectivity = connectivity.at[:, np.arange(1,edex_max*dim, 2)].set(Elemental_patch_nodes*2+1)
    return connectivity

def get_A_b_CFEM(XY, XY_host, Elem_nodes, Elem_nodes_host, Gauss_Num_CFEM, quad_num_CFEM, dim, elem_type, nodes_per_elem, dof_global, 
                 Cmat, Elem_tr, nelem_tr, Gauss_Num_tr, connectivity_tr, iffix,
                 indices, indptr, s_patch, d_c, edex_max, ndex_max, a_dil, mbasis):
    
    # decide how many blocks are we gonnna use
    start_time = time.time()
    size_BTB = int(nelem) * int(Gauss_Num_CFEM**dim) * int(edex_max*dim) * int(edex_max*dim)
    size_Phi = int(nelem) * int(Gauss_Num_CFEM**dim) * int(elem_dof) * int(edex_max) * int(1+dim)
    size_Gs  = int(nelem) * int(Gauss_Num_CFEM**dim) * int(elem_dof) * int((ndex_max+mbasis)**2)
    nblock = int(max(size_BTB, size_Phi, size_Gs) // max_array_size_block + 1)
    nelem_per_block_regular = nelem // nblock
    if nelem % nblock == 0:
        nelem_per_block_remainder = nelem_per_block_regular
    else:
        nelem_per_block_remainder = nelem_per_block_regular + nelem % nblock
    print(f"CFEM A_sp -> {nblock} blocks")
    
    shape_vals = get_shape_vals(Gauss_Num_CFEM, dim, elem_type) # (num_quads, num_nodes)     
    for iblock in range(nblock):
        if iblock == nblock-1:
            nelem_per_block = nelem_per_block_remainder
            elem_idx_block = np.array(range(nelem_per_block_regular*iblock, nelem_per_block_regular*iblock 
                                            + nelem_per_block_remainder), dtype=np.int32)
        else:
            nelem_per_block = nelem_per_block_regular
            elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
        
        Elem_nodes_block = Elem_nodes[elem_idx_block]
        shape_grads_physical_block, JxW_block = get_shape_grads(Gauss_Num_CFEM, dim, elem_type, XY, Elem_nodes_block)    
        # print('JxW', JxW_block.shape)
        
        (N_til_block, Grad_N_til_block, JxW_block, 
         Elemental_patch_nodes_st_block, Elem_nodes_block) = get_CFEM_shape_fun_block(elem_idx_block, nelem_per_block,
                   XY, XY_host, Elem_nodes, Elem_nodes_host, shape_vals, Gauss_Num_CFEM, quad_num_CFEM, dim, elem_type, nodes_per_elem,
                   indices, indptr, s_patch, d_c, edex_max, ndex_max,
                    a_dil, mbasis)
        connectivity_block = get_connectivity(Elemental_patch_nodes_st_block, dim)
        # print(Elemental_patch_nodes_st_block)
        if iblock == 0:
            jit_time = time.time() - start_time
            print(f"CFEM shape function jit compile took {jit_time:.4f} seconds")
        
        
        Bmat_block = np.zeros((nelem_per_block, quad_num_CFEM, 3, edex_max*dim), dtype=np.double)
        Bmat_block = Bmat_block.at[:,:,0,np.arange(0,edex_max*dim, 2)].set(Grad_N_til_block[:,:,:,0])
        Bmat_block = Bmat_block.at[:,:,1,np.arange(1,edex_max*dim, 2)].set(Grad_N_til_block[:,:,:,1])
        Bmat_block = Bmat_block.at[:,:,2,np.arange(1,edex_max*dim, 2)].set(Grad_N_til_block[:,:,:,0])
        Bmat_block = Bmat_block.at[:,:,2,np.arange(0,edex_max*dim, 2)].set(Grad_N_til_block[:,:,:,1])    
        
        BTB_block = np.matmul(np.transpose(Bmat_block, (0,1,3,2)), Cmat[None,None,:,:]) # (num_cells, num_quads, num_nodes, num_nodes)
        BTB_block = np.matmul(BTB_block, Bmat_block)
        V_block = np.sum(BTB_block * JxW_block[:, :, None, None], axis=1).reshape(-1) # (num_cells, num_nodes, num_nodes) -> (1 ,)
        I_block = np.repeat(connectivity_block, edex_max*dim, axis=1).reshape(-1)
        J_block = np.repeat(connectivity_block, edex_max*dim, axis=0).reshape(-1)
        # print(V_block.shape, I_block.shape, J_block.shape)
        
        
        if iblock == 0:
            A_sp_scipy =  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
        else:
            A_sp_scipy +=  csc_matrix((V_block, (I_block, J_block)), shape=(dof_global, dof_global)) 
        # print(A_sp_scipy.toarray()[:12,:12]) 
    print(f'CFEM A_sp took {time.time() - start_time:.4f} seconds')
    
    

    # Traction force
    start_time = time.time()
    rhs = np.zeros(dof_global, dtype=np.double)
    Elem_nodes_tr_full = Elem_nodes[Elem_tr] # full nodes
    size_BTB = int(nelem_tr) * int(Gauss_Num_CFEM) * int(edex_max*dim) * int(edex_max*dim)
    size_Phi = int(nelem_tr) * int(Gauss_Num_CFEM) * int(elem_dof) * int(edex_max) * int(1+dim)
    size_Gs  = int(nelem_tr) * int(Gauss_Num_CFEM) * int(elem_dof) * int((ndex_max+mbasis)**2)
    nblock = int(max(size_BTB, size_Phi, size_Gs) // max_array_size_block + 1)
    nelem_per_block_regular = nelem_tr // nblock
    if nelem_tr % nblock == 0:
        nelem_per_block_remainder = nelem_per_block_regular
    else:
        nelem_per_block_remainder = nelem_per_block_regular + nelem_tr % nblock
    print(f"CFEM rhs (or b) -> {nblock} blocks")
    
    shape_vals = get_shape_vals(Gauss_Num_CFEM, 1, elem_type) # dim == 1, linear element ()
    if elem_type == 'CPE4': # recover the original dimension
        shape_vals = np.concatenate((np.zeros_like(shape_vals), shape_vals), axis=1)
    if elem_type == 'CPE3': # recover the original dimension
        shape_vals = np.concatenate((shape_vals, np.zeros((len(shape_vals),1)), ), axis=1)
    # print('Traction shape_vals\n', shape_vals)
    # shape_vals = get_shape_vals_tr(Gauss_Num_CFEM, elem_type, norm_vec) # (Gauss_Num_CFEM, num_nodes), for traction boundary  
    for iblock in range(nblock):
        if iblock == nblock-1:
            nelem_per_block = nelem_per_block_remainder
            elem_idx_block = np.array(range(nelem_per_block_regular*iblock, nelem_per_block_regular*iblock 
                                            + nelem_per_block_remainder), dtype=np.int32)
        else:
            nelem_per_block = nelem_per_block_regular
            elem_idx_block = np.array(range(nelem_per_block*iblock, nelem_per_block*(iblock+1)), dtype=np.int32)
        
        Elem_nodes_block = Elem_nodes_tr_full[elem_idx_block] # full nodes
        Elem_nodes_block_host = onp.array(Elem_nodes_block)
        Elem_tr_block = Elem_tr[elem_idx_block]
        
        _, JxW = get_shape_grads(Gauss_Num_tr, 1, elem_type, XY[:,[1]], Elem_nodes_tr) # Elem_nodes_tr: 1D 2 nodes
        # print('tr', JxW)
        # print(Elem_nodes_block)
        # print(Elem_tr_block)
        
        # start_patch = time.time()
        (Elemental_patch_nodes_st_block, edexes_block,
         Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, 
         ndexes_block) = get_patch_info(indices, indptr, s_patch, d_c, edex_max, ndex_max, XY_host, Elem_nodes_block_host, 
                                        nelem_per_block, nodes_per_elem)   
        # print(Elemental_patch_nodes_st_block)
        # print(Nodal_patch_nodes_st_block)
        
        
        #############  get G #############
        # start_G = time.time()
        # compute moment matrix for the meshfree shape functions
        vmap_inputs_G = np.concatenate((np.repeat(np.arange(nelem_per_block), nodes_per_elem), # ielem
                          np.tile(np.arange(nodes_per_elem), nelem_per_block))).reshape(2,-1).T # inode
        Gs = jax.vmap(get_Gs, in_axes = (0, 
                None, None, None, None, 
                None, None, None), out_axes = 0
                )(vmap_inputs_G,
                  Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, ndexes_block, ndex_max,
                  XY, a_dil, mbasis) 
        Gs = np.reshape(Gs, (nelem_per_block, nodes_per_elem, ndex_max+mbasis, ndex_max+mbasis))
        
        ############# Phi ###############
        # linearize vmap inputs (ielem, iquad, inode)
        vmap_inputs = np.concatenate((np.repeat(np.arange(nelem_per_block), Gauss_Num_tr*nodes_per_elem), # ielem_idx
                          np.tile(np.repeat(np.arange(Gauss_Num_tr), nodes_per_elem), nelem_per_block), # iquad
                          np.tile(np.arange(nodes_per_elem), nelem_per_block*Gauss_Num_tr))).reshape(3,-1).T # inode
        
        # Vmap for CFEM shape functions for regular elements
        Phi = jax.vmap(get_phi, in_axes = (0, None, None, None, None, 
                None, None, None, None, None, 
                None, None, None, None), out_axes = 0
                )(vmap_inputs, Elem_tr_block, Gs, shape_vals, edex_max,
                Nodal_patch_nodes_st_block, Nodal_patch_nodes_bool_block, Nodal_patch_nodes_idx_block, ndexes_block, ndex_max,
                XY, Elem_nodes, a_dil, mbasis)
        Phi = np.reshape(Phi, (nelem_per_block, Gauss_Num_tr, nodes_per_elem, edex_max, 1+dim))
        N_til = np.sum(shape_vals[None, :, :, None]*Phi[:,:,:,:,0], axis=2) # (num_cells, num_quads, edex_max)
        # print(Phi[:,0,0,:,0])
        N_trac = np.zeros((nelem_per_block, Gauss_Num_tr, edex_max*dim, dim), dtype=np.double) # (nelem_tr, quad_num_tr, elem_dof, dim)
        N_trac = N_trac.at[:,:,np.arange(0,edex_max*dim, 2),0].set(N_til[:,:,:]) # column 0
        N_trac = N_trac.at[:,:,np.arange(1,edex_max*dim, 2),1].set(N_til[:,:,:]) # column 1
        
        connectivity = get_connectivity(Elemental_patch_nodes_st_block, dim)
        NP = np.sum( np.matmul(N_trac, traction_force[None,None,:,:]) * JxW[:,:,None,None], axis=1) # (nelem_tr, elem_dof, 1)
        rhs = rhs.at[connectivity.reshape(-1)].add(NP.reshape(-1))  # assemble 
        # print(rhs)
    print(f'CFEM rhs (or b) took {time.time() - start_time:.4f} seconds')
    
    # Apply boundary condition - Penalty method
    for idx, fix in enumerate(iffix):
        if fix:
            A_sp_scipy[idx,idx] *= 1e10
            rhs.at[idx].set(0)
            
    return A_sp_scipy, rhs, jit_time


#%% Solver

@partial(jax.jit, static_argnames=['nchunk','dof_per_chunk','dof_per_chunk_remainder'])
def get_residual(sol, A_sp, b, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    res = b - A_sp @ sol
    res = res.at[inds_nodes_list].set(0) # disp. BC
    return res

@partial(jax.jit, static_argnames=['nchunk','dof_per_chunk','dof_per_chunk_remainder'])
def get_Ap(p, A_sp, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    Ap = A_sp @ p
    Ap = Ap.at[inds_nodes_list].set(0) # disp. BC
    return Ap

def get_residual_chunks(sol, A_sp_scipy, b, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    
    res = np.zeros(dof_global, dtype=np.double)  
    for ichunk in range(nchunk):
        if ichunk < nchunk-1:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_per_chunk*(ichunk+1)), dtype=np.int32)
        else:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_global), dtype=np.int32)
        A_sp_chunk = BCOO.from_scipy_sparse(A_sp_scipy[dof_idx_chunk,:])
        res_chunk = b[dof_idx_chunk] - A_sp_chunk @ sol
        res = res.at[dof_idx_chunk].set(res_chunk)
    res = res.at[inds_nodes_list].set(0) # disp. BC
    return res
    
def get_Ap_chunks(p, A_sp_scipy, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder):
    
    Ap = np.zeros(dof_global, dtype=np.double)  
    for ichunk in range(nchunk):
        if ichunk < nchunk-1:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_per_chunk*(ichunk+1)), dtype=np.int32)
        else:
            dof_idx_chunk = np.array(range(dof_per_chunk*ichunk, dof_global), dtype=np.int32)
        A_sp_chunk = BCOO.from_scipy_sparse(A_sp_scipy[dof_idx_chunk,:])
        Ap_chunk = A_sp_chunk @ p
        Ap = Ap.at[dof_idx_chunk].set(Ap_chunk)
    Ap = Ap.at[inds_nodes_list].set(0) # disp. BC
    return Ap

def CG_solver(get_residual, get_Ap, sol, A_sp, b, inds_nodes_list, dof_global, tol, 
              nchunk, dof_per_chunk, dof_per_chunk_remainder):
    start_time = time.time()
    r = get_residual(sol, A_sp, b, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder)
    p = r
    rsold = np.dot(r,r)
    for step in range(dof_global*100):
        # print(p)
        Ap = get_Ap(p, A_sp, inds_nodes_list, nchunk, dof_per_chunk, dof_per_chunk_remainder)
        alpha = rsold / np.dot(p, Ap)
        sol += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r,r)
        
        if step%10000 == 0:
            print(f"step = {step}, res l_2 = {rsnew**0.5}") 
        
        if rsnew**0.5 < tol:
            break
        p = r + (rsnew/rsold) * p
        rsold = rsnew
    print(f"CG solver took {time.time() - start_time:.4f} seconds\n")
    return sol


#%% Main program - Cubic spline functions

############# 2D FEM Cooks beam ###############


gpu_idx = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

# Problem settings
p_dict={0:0, 1:3, 2:6, 3:10, 4:15, 5:21, 6:28, 7:36, 8:45, 9:55, 10:66} # number of complete basis for polynomial order p
# ps = [4]      # [0, 1, 2, 3]
alpha_dils = [40]       # [1.2, 1.4, 2.2, 2.4, 3.2, 3.4, 4.2, 4.4]    # dilation parameter
# alpha_dils = onp.arange(1, 6.2, 0.2)
s_patches = [1]         # elemental patch_size
ps = [-1] # -1 : same as s_patch
nelems = [4]        # [4, 10, 15, 20, 30, 40, 50, 100, 200, 400, 700, 1024]

# max_array_size_block = 2e7  # 5e8 for Athena / 2e7 for laptop
# max_array_size_chunk = 2e7  #
GPUs = GPUtil.getGPUs()
if len(GPUs) > 1:
    max_array_size_block = 4e8  # 2e9 for Athena
    max_array_size_chunk = 1e9  #
else:
    max_array_size_block = 2e7  # 2e8 for laptop
    max_array_size_chunk = 2e7  #

elem_types = ['CPE4'] #  'CPE4', 'CPE8', 'CPE3', 'CPE6'
dim = 2

# material properties
P = 100/16 # traction force
traction_force = np.array([[0], [P]], dtype=np.float64)
# traction_force = onp.array([[P], [0]], dtype=onp.float64)
L, D = 48, 60
E = 250
nu = 0.4999
# nu = 0.3
Cmat = np.array([[1-nu, nu, 0], 
                 [nu, 1-nu, 0], 
                 [0, 0, (1-2*nu)/2]], dtype=np.double) * E / ((1+nu)*(1-2*nu))
parent_dir = os.path.abspath(os.getcwd())

run_FEM = True
run_CFEM = False
radial_patch_bool = False
regular_mesh_bool = True
export_output = False

export_terminal = False
if export_terminal:
    sys.stdout = open("main32_comp_time_CPU.txt", "w")

#%%

for elem_type in elem_types:
    nodes_per_elem = int(elem_type[3])
    elem_dof = nodes_per_elem * dim    
    
    # define quadrature numbers
    
    if elem_type == 'CPE4': # linear quad element
        Gauss_Num_FEM = 2   # 4 
        Gauss_Num_CFEM = 6  # 6
        Gauss_Num_tr = 6
    elif elem_type == 'CPE8': # quadratic quad element
        Gauss_Num_FEM = 4   # 4 
        Gauss_Num_CFEM = 6  # 6
        Gauss_Num_tr = 6
    elif elem_type == 'CPE3':# triangular element
        Gauss_Num_FEM = 4   # 1, 3, 4, 6 
        Gauss_Num_CFEM = 6  # 1, 3, 4, 6
        Gauss_Num_tr = 6    # this is for CFEM line integral
    elif elem_type == 'CPE6': # triangular element
        Gauss_Num_FEM = 3   # 1, 3, 4, 6 
        Gauss_Num_CFEM = 6  # 1, 3, 4, 6
        Gauss_Num_tr = 4    # this is always for FEM quadratic integral
    
    
    if elem_type == 'CPE4' or elem_type == 'CPE8':
        quad_num_FEM = Gauss_Num_FEM ** dim
        quad_num_CFEM = Gauss_Num_CFEM ** dim
    elif elem_type == 'CPE3' or elem_type == 'CPE6':
        quad_num_FEM = Gauss_Num_FEM
        quad_num_CFEM = Gauss_Num_CFEM
    
    for s_patch in s_patches:
        for p in ps: # polynomial orders
            if p == -1:
                p = s_patch
        
            mbasis = p_dict[p]  
            for alpha_dil in alpha_dils:
                for nelem_y in nelems:
                    nelem_x = nelem_y
                    
                    
                    #FEM settings
                    input_file_name = str(nelem_y) + '_' + elem_type + '.inp'
                    parent_dir = os.path.abspath(os.getcwd())
                    problem_type = '2D_Cook'                    
                    XY_host, Elem_nodes_host, iffix_host, connectivity_host, Elem_tr_host, P_, norm_vec = read_mesh_ABQ(parent_dir, problem_type, input_file_name)
                    nnode = len(XY_host)
                    dof_global = len(iffix_host)
                    nelem = len(Elem_nodes_host)
                    
                    # re-define boundary elements
                    if elem_type == 'CPE4' or elem_type == 'CPE3':
                        Elem_nodes_tr = onp.zeros((len(Elem_tr_host), 2), dtype=onp.int64)  # (nelem_tr, 2 nodes)
                        connectivity_tr = onp.zeros((len(Elem_tr_host), 4), dtype=onp.int64)
                    elif elem_type == 'CPE8'or elem_type == 'CPE6':
                        Elem_nodes_tr = onp.zeros((len(Elem_tr_host), 3), dtype=onp.int64)
                        connectivity_tr = onp.zeros((len(Elem_tr_host), 6), dtype=onp.int64)
                    for ielem_idx, ielem in enumerate(Elem_tr_host):
                        elem_nodes = Elem_nodes_host[ielem]
                        XY_elem = onp.concatenate((onp.expand_dims(elem_nodes,axis=1), XY_host[elem_nodes]), axis=1)
                        XY_elem_tr = XY_elem[onp.where(XY_elem[:,1]==L)[0]] # check if x coordinate is at the boundary
                        elem_nodes_tr = XY_elem_tr[onp.argsort(XY_elem_tr[:,2]), 0]
                        Elem_nodes_tr[ielem_idx, :] = elem_nodes_tr
                        for idx, inode in enumerate(elem_nodes_tr):
                            connectivity_tr[ielem_idx, 2*idx] = 2*inode
                            connectivity_tr[ielem_idx, 2*idx+1] = 2*inode+1
                    Elem_nodes_tr = np.array(Elem_nodes_tr)
                    connectivity_tr = np.array(connectivity_tr)
                        
                    # host to device array
                    XY = np.array(XY_host)
                    Elem_nodes = np.array(Elem_nodes_host)
                    iffix = np.array(iffix_host)
                    connectivity = np.array(connectivity_host)
                    Elem_tr = np.array(Elem_tr_host)
                    nelem_tr = len(Elem_tr)
    
                    # mem_report(1, gpu_idx)
    
                    if run_FEM == True:
                        print(f"\n--------- FEM {elem_type} nelem_x: {nelem_x} DOFs: {dof_global} ----------")
            
                        # compute FEM basic stuff
                        start_time = time.time()
                        A_sp_scipy, b = get_A_b_FEM(XY, Elem_nodes, connectivity, nelem, Gauss_Num_FEM, quad_num_FEM, dim, elem_type, elem_dof, dof_global, 
                                                    Cmat, Elem_nodes_tr, nelem_tr, Gauss_Num_tr, connectivity_tr, iffix)
                        print(f"FEM A and b took {time.time() - start_time:.4f} seconds")
                        
                        
                        # CG solver
                        # Divide A_sp matrix into nchunk
                        size_A_sp = dof_global * 9 * 3 # 9 is nodal connectivity. Each node is connected to 9 surrounding nodes. 3 means sparse values & indicies 
                        nchunk = int(size_A_sp // max_array_size_chunk + 1)
                        if dof_global % nchunk == 0:
                            dof_per_chunk = dof_global // nchunk
                            dof_per_chunk_remainder = dof_per_chunk
                        else:
                            dof_per_chunk = dof_global // nchunk
                            dof_per_chunk_remainder = dof_per_chunk + dof_global % nchunk
                        print(f"A_sp array -> {nchunk} chunks")
                        
                        inds_nodes_list = np.where(iffix==1)[0]
                        sol = np.zeros(dof_global, dtype=np.double)              # (dof,)
                        tol = 1e-10
                        
                        if nchunk == 1: # when DOFs are small
                            A_sp = BCOO.from_scipy_sparse(A_sp_scipy)
                            sol = CG_solver(get_residual, get_Ap, sol, A_sp, b, inds_nodes_list, dof_global, tol, 
                                            nchunk, dof_per_chunk, dof_per_chunk_remainder)
                        else: # when DOFs are big
                            sol = CG_solver(get_residual_chunks, get_Ap_chunks, sol, A_sp_scipy, b, inds_nodes_list, dof_global, tol, 
                                            nchunk, dof_per_chunk, dof_per_chunk_remainder)
                        u_glo = onp.array(sol)
                        xy = onp.zeros_like(XY_host)# compare
                        uv = onp.zeros_like(XY_host)
                        for inode, iXY in enumerate(XY_host):
                            xy[inode,:] = iXY + u_glo[inode*2:inode*2+2]
                            uv[inode,:] = u_glo[inode*2:inode*2+2]
                        print(f'v_h: {uv[nelem_x*(nelem_x+1),1]: .4e}')
                        # print(uv)
                       
                        mem_report(2, gpu_idx)
    
                    #%% ########################## CFEM ######################
                    if run_CFEM == False:
                        continue
                    if elem_type != 'CPE4' and elem_type != 'CPE3':
                        continue
                    if s_patch*2 > nelem_x:
                        continue
                    if p > s_patch or p > alpha_dil:
                        continue
                    # compute adjacency matrix - Serial
                    print(f"\n- - - - - - CFEM {elem_type} nelem_x: {nelem_x} with s: {s_patch}, a: {alpha_dil}, p: {p} - - - - - -")  
                    
                    start_time_org = time.time()
                    indices, indptr = get_adj_mat(Elem_nodes_host, nnode, s_patch)
                    print(f"CFEM adj_s matrix took {time.time() - start_time_org:.4f} seconds")
            
                    # # patch settings
                    d_c = L/nelem_y     # characteristic length in physical coord.
                    a_dil = alpha_dil * d_c
                    
                    # compute Elemental patch - Serial
                    start_time = time.time()
                    edex_max, ndex_max = get_dex_max(indices, indptr, s_patch, d_c, XY_host, Elem_nodes_host, nelem, nnode, nodes_per_elem)
                    print(f'edex_max / ndex_max: {edex_max} / {ndex_max}, took {time.time() - start_time:.4f} seconds')
                    
                    ############################  get_A_b_CFEM function ############
                    start_time = time.time()
                    A_sp_scipy, b, jit_time = get_A_b_CFEM(XY, XY_host, Elem_nodes, Elem_nodes_host, Gauss_Num_CFEM, quad_num_CFEM, dim, elem_type, nodes_per_elem, dof_global, 
                                                    Cmat, Elem_tr, nelem_tr, Gauss_Num_tr, connectivity_tr, iffix,
                                                    indices, indptr, s_patch, d_c, edex_max, ndex_max, a_dil, mbasis)
                    print(f"CFEM A and b took {time.time() - start_time:.4f} seconds")
                    mem_report(3, gpu_idx)
                    
                    # CG solver
                    # Divide A_sp matrix into nchunk
                    size_A_sp = dof_global * edex_max * 3 # edex_max is nodal connectivity ~ bandwidth. 3 means sparse values & indicies 
                    nchunk = int(size_A_sp // max_array_size_chunk + 1)
                    if dof_global % nchunk == 0:
                        dof_per_chunk = dof_global // nchunk
                        dof_per_chunk_remainder = dof_per_chunk
                    else:
                        dof_per_chunk = dof_global // nchunk
                        dof_per_chunk_remainder = dof_per_chunk + dof_global % nchunk
                    print(f"A_sp array -> {nchunk} chunks")
                    
                    inds_nodes_list = np.where(iffix==1)[0]
                    sol = np.zeros(dof_global, dtype=np.double)              # (dof,)
                    tol = 1e-10
                    
                    if nchunk == 1: # when DOFs are small
                        A_sp = BCOO.from_scipy_sparse(A_sp_scipy)
                        sol = CG_solver(get_residual, get_Ap, sol, A_sp, b, inds_nodes_list, dof_global, tol, 
                                        nchunk, dof_per_chunk, dof_per_chunk_remainder)
                    else: # when DOFs are big
                        sol = CG_solver(get_residual_chunks, get_Ap_chunks, sol, A_sp_scipy, b, inds_nodes_list, dof_global, tol, 
                                        nchunk, dof_per_chunk, dof_per_chunk_remainder)
                    u_glo = onp.array(sol)
                    xy = onp.zeros_like(XY_host)# compare
                    uv = onp.zeros_like(XY_host)
                    for inode, iXY in enumerate(XY_host):
                        xy[inode,:] = iXY + u_glo[inode*2:inode*2+2]
                        uv[inode,:] = u_glo[inode*2:inode*2+2]
                    print(f'v_h: {uv[nelem_x*(nelem_x+1),1]: .4e}')
                    # print(uv)
                        
                    # mem_report(4, gpu_idx)
        
        

# if export_terminal:
#     sys.stdout.close()

