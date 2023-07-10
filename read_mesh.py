import numpy as np
import os
import time

# Read mesh from abaqus input file
# written by Chanwook Park (chanwookpark2024@u.northwestern.edu)

def read_mesh_ABQ(parent_dir, problem_type, input_file_name):
    
    # start_time = time.time()
    
    path = os.path.join(parent_dir, 'abaqus_input')
    path = os.path.join(path, problem_type)
    path = os.path.join(path, input_file_name)
    file = open(path, 'r')
    lines = file.readlines()
    list_XY, list_elem_nodes = [], []
    for count, line in enumerate(lines):
        if '*Node' in line:
            count += 1
            node_bool = True
            while node_bool:
                if '*Element' in lines[count+1]:
                    node_bool = False
                # print(lines[count])
                line_list =  [float(item.strip()) for item in lines[count].strip().split(',')] 
                list_XY.append(line_list[1:])
                count += 1
            XY = np.array(list_XY)
            # print(f'---XY took {time.time() - start_time:.4f} seconds')
            # start_time = time.time()
        
        if '*Element' in line:
            count += 1
            node_bool = True
            while node_bool:
                if '*Nset' in lines[count+1]:
                    node_bool = False
                line_list =  [int(item.strip()) for item in lines[count].strip().split(',')] 
                # line_list_order = line_list[2:]
                # line_list_order.append(line_list[1]) # the way of how we circulate an element
                list_elem_nodes.append(line_list[1:])
                count += 1
            elem_nodes = np.array(list_elem_nodes) - 1 # for python numbering
            # print(f'---Elem_nodes took {time.time() - start_time:.4f} seconds')
            # start_time = time.time()
        
        # # read traction bc set
        # if '*Elset' in line and 'Surf-1_S' in line and 'instance' in line:
        #     count += 1
        #     if 'generate' in line: # when the fixed boundary nodes are represented as a 'generate' mode
        #         line_list =  [int(item.strip()) for item in lines[count].strip().split(',')]
        #         # print(line_list)
        #         # elements at traction boundary
        #         Elem_trac = np.linspace(line_list[0], line_list[1], 
        #                                         int( (line_list[1]-line_list[0])/line_list[2]) +1, dtype=np.int64) - 1
                # print(Elem_trac)
        # if 'Surface traction' in line:
        #     count += 2
        #     line_list =  [item.strip() for item in lines[count].strip().split(',')]
        #     P = float(line_list[2])
        #     norm_vec = [float(item) for item in line_list[3:]]
           
            
            # else:
            #     nset_bool = True
            #     line_list = []
            #     while nset_bool:
            #         if '*Elset' in lines[count+1]:
            #             nset_bool = False
            #         line_list +=  [int(item.strip()) for item in lines[count].strip().split(',')] 
            #         count += 1
            #     fixed_node = np.array(line_list) - 1 # for python numbering
            # # print(count)
            continue
        
    # print(f'---Read Abaqus input took {time.time() - start_time:.4f} seconds')
    # start_time = time.time()
    
    
    # iffix
    iffix = np.zeros(XY.shape[0]*XY.shape[1])
    count = 0
    for row in range(XY.shape[0]):
        for col in range(XY.shape[1]):
            if XY[row, 0] == 0:
                # print(row, count)
                iffix[count] = 1;
            count += 1 # global dof order: [u1, v1, u2, v2, ...].T
    
    # print(f'---iffix took {time.time() - start_time:.4f} seconds')
    # start_time = time.time()
    
    
    # connectivity
    nelem = elem_nodes.shape[0]
    dimension = 2
    for col in range(elem_nodes.shape[1]):
        # append global DOF for u
        if col == 0:
            connectivity = elem_nodes[:,0].reshape(nelem, 1) * dimension
        else:
            connectivity = np.append(connectivity, elem_nodes[:,col].reshape(nelem, 1) * dimension, axis=1)
        
        # append global DOF for v
        connectivity = np.append(connectivity, elem_nodes[:,col].reshape(nelem, 1) * dimension + 1, axis=1)
    
    # print(f'---connectivity input took {time.time() - start_time:.4f} seconds')
    
    
    return XY, elem_nodes, iffix, connectivity

def read_mesh_ABQ_2D_Beam(parent_dir, problem_type, input_file_name):
    
    # start_time = time.time()
    
    path = os.path.join(parent_dir, 'abaqus_input')
    path = os.path.join(path, problem_type)
    path = os.path.join(path, input_file_name)
    file = open(path, 'r')
    lines = file.readlines()
    list_XY, list_elem_nodes = [], []
    for count, line in enumerate(lines):
        if '*Node' in line and 'Beam' in lines[count-1]:
            count += 1
            node_bool = True
            while node_bool:
                if '*Element' in lines[count+1]:
                    node_bool = False
                # print(lines[count])
                line_list =  [float(item.strip()) for item in lines[count].strip().split(',')] 
                list_XY.append(line_list[1:])
                count += 1
            XY = np.array(list_XY)
            # print(f'---XY took {time.time() - start_time:.4f} seconds')
            # start_time = time.time()
        
        if '*Element' in line and 'CPS4' in line:
            count += 1
            node_bool = True
            while node_bool:
                if '*Nset' in lines[count+1]:
                    node_bool = False
                # print(lines[count])
                line_list =  [int(item.strip()) for item in lines[count].strip().split(',')] 
                # line_list_order = line_list[2:]
                # line_list_order.append(line_list[1]) # the way of how we circulate an element
                list_elem_nodes.append(line_list[1:])
                count += 1
            elem_nodes = np.array(list_elem_nodes) - 1 # for python numbering
            # print(f'---Elem_nodes took {time.time() - start_time:.4f} seconds')
            # start_time = time.time()
        
            continue
        
    # print(f'---Read Abaqus input took {time.time() - start_time:.4f} seconds')
    # start_time = time.time()
    
    
    # iffix
    iffix = np.zeros(XY.shape[0]*XY.shape[1])
    
    
    # connectivity
    nelem = elem_nodes.shape[0]
    dimension = 2
    for col in range(elem_nodes.shape[1]):
        # append global DOF for u
        if col == 0:
            connectivity = elem_nodes[:,0].reshape(nelem, 1) * dimension
        else:
            connectivity = np.append(connectivity, elem_nodes[:,col].reshape(nelem, 1) * dimension, axis=1)
        
        # append global DOF for v
        connectivity = np.append(connectivity, elem_nodes[:,col].reshape(nelem, 1) * dimension + 1, axis=1)
    
    # print(f'---connectivity input took {time.time() - start_time:.4f} seconds')
    
    
    return XY, elem_nodes, iffix, connectivity

def read_mesh_ABQ_3D(parent_dir, input_file_name, elem_type):
    
    # start_time = time.time()
    
    path = os.path.join(parent_dir, 'abaqus_input')
    path = os.path.join(path, '3D_BlockUnderCompression')
    path = os.path.join(path, input_file_name)
    file = open(path, 'r')
    lines = file.readlines()
    list_XY, list_elem_nodes = [], []
    for count, line in enumerate(lines):
        if '*Node' in line and 'Nset' not in lines[count+2]: # distinguish from reference point node
            count += 1
            node_bool = True
            while node_bool:
                if '*Element' in lines[count+1]:
                    node_bool = False
                line_list =  [float(item.strip()) for item in lines[count].strip().split(',')] 
                list_XY.append(line_list[1:])
                count += 1
            XY = np.array(list_XY)
            # print(f'---XY took {time.time() - start_time:.4f} seconds')
            # start_time = time.time()
        
        if '*Element' in line:
            count += 1
            node_bool = True
            while node_bool:
                if ('*Nset' in lines[count+1] and elem_type != 'C3D20') or (
                        '*Nset' in lines[count+2] and elem_type == 'C3D20'):
                    node_bool = False
                # print(lines[count])
                # print(lines[count][-2:])
                if lines[count][-2] == ',':
                    # print(lines[count] + lines[count+1])
                    line_elem = lines[count] + lines[count+1]
                    # line_list = [int(item.strip()) for item in line_elem.strip().split(',')]
                    count += 2
                else:
                    # print(lines[count])
                    line_elem = lines[count]
                    count += 1
                line_list = line_list = [int(item.strip()) for item in line_elem.strip().split(',')]
                
                # line_list =  [int(item.strip()) for item in lines[count].strip().split(',')] 
                # line_list_order = line_list[2:]
                # line_list_order.append(line_list[1]) # the way of how we circulate an element
                list_elem_nodes.append(line_list[1:])
                # count += 1
            elem_nodes = np.array(list_elem_nodes) - 1 # for python numbering
            # print(f'---Elem_nodes took {time.time() - start_time:.4f} seconds')
            # start_time = time.time()
        
            continue
    
    # iffix
    iffix = np.zeros(XY.shape[0]*XY.shape[1])
    count = 0
    for row in range(XY.shape[0]):
        for col in range(XY.shape[1]):
            if XY[row, 2] == 0: # z coord is zero
                # print(row, count)
                iffix[count] = 1;
            count += 1 # global dof order: [u1, v1, w1, u2, v2, w2, ...].T
    
    # print(f'---iffix took {time.time() - start_time:.4f} seconds')
    # start_time = time.time()
    
    
    # connectivity
    nelem = elem_nodes.shape[0]
    dimension = 3
    for col in range(elem_nodes.shape[1]):
        # append global DOF for u
        if col == 0:
            connectivity = elem_nodes[:,0].reshape(nelem, 1) * dimension
        else:
            connectivity = np.append(connectivity, elem_nodes[:,col].reshape(nelem, 1) * dimension, axis=1)
        
        # append global DOF for v
        connectivity = np.append(connectivity, elem_nodes[:,col].reshape(nelem, 1) * dimension + 1, axis=1)
        
        # append global DOF for w
        connectivity = np.append(connectivity, elem_nodes[:,col].reshape(nelem, 1) * dimension + 2, axis=1)
    
    
    # print(f'---connectivity input took {time.time() - start_time:.4f} seconds')
    
    
    return XY, elem_nodes, iffix, connectivity