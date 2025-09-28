import numpy as np
from placedb import RefineDB
import numba
from numba import cuda
import math

@cuda.jit
def cal_e_plus_minus(e_plus, e_minus, pin_pos, pin2net):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for pin_idx in range(idx_start, pin_pos.shape[0], threads_per_grid):
        net_idx = pin2net[pin_idx]
        cuda.atomic.max(e_plus, net_idx, pin_pos[pin_idx])
        cuda.atomic.min(e_minus, net_idx, pin_pos[pin_idx])

@cuda.jit
def cal_a_plus_minus(a_plus, a_minus, e_plus, e_minus, pin_pos, pin2net, gamma):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for pin_idx in range(idx_start, pin_pos.shape[0], threads_per_grid):
        net_idx = pin2net[pin_idx]
        a_plus[pin_idx] = math.exp((pin_pos[pin_idx] - e_plus[net_idx]) / gamma)
        a_minus[pin_idx] = math.exp(-(pin_pos[pin_idx] - e_minus[net_idx]) / gamma)

@cuda.jit
def cal_b_plus_minus(b_plus, b_minus, a_plus, a_minus, pin2net):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for pin_idx in range(idx_start, pin2net.shape[0], threads_per_grid):
        net_idx = pin2net[pin_idx]
        cuda.atomic.add(b_plus, net_idx, a_plus[pin_idx])
        cuda.atomic.add(b_minus, net_idx, a_minus[pin_idx])

@cuda.jit
def cal_c_plus_minus(c_plus, c_minus, a_plus, a_minus, pin_pos, pin2net):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for pin_idx in range(idx_start, pin2net.shape[0], threads_per_grid):
        net_idx = pin2net[pin_idx]
        cuda.atomic.add(c_plus, net_idx, a_plus[pin_idx] * pin_pos[pin_idx])
        cuda.atomic.add(c_minus, net_idx, a_minus[pin_idx] * pin_pos[pin_idx])

@cuda.jit
def cal_tol_wl(wl, b_plus, b_minus, c_plus, c_minus):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for net_idx in range(idx_start, b_plus.shape[0], threads_per_grid):
        term = c_plus[net_idx] / b_plus[net_idx] - c_minus[net_idx] / b_minus[net_idx]
        cuda.atomic.add(wl, 0, term)

@cuda.jit
def cal_dpos(dpos, a_plus, a_minus, b_plus, b_minus, c_plus, c_minus, pin_pos, pin2net, pin2node, gamma, pin_cnt):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for pin_idx in range(idx_start, pin2net.shape[0], threads_per_grid):
        if pin_idx < pin_cnt:
            net_idx, node_idx = pin2net[pin_idx], pin2node[pin_idx]
            term = a_plus[pin_idx] * ((1 + pin_pos[pin_idx] / gamma) * b_plus[net_idx] - c_plus[net_idx] / gamma) / b_plus[net_idx] ** 2
            term -= a_minus[pin_idx] * ((1 - pin_pos[pin_idx] / gamma) * b_minus[net_idx] + c_minus[net_idx] / gamma) / b_minus[net_idx] ** 2
            cuda.atomic.add(dpos, node_idx, term)

@cuda.jit
def cal_pin_pos(pin_pos, pin_offset, node_pos, pin2node, pin_cnt):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for pin_idx in range(idx_start, pin_offset.shape[0], threads_per_grid):
        if pin_idx < pin_cnt:
            node_idx = pin2node[pin_idx]
            pin_pos[pin_idx] = pin_offset[pin_idx] + node_pos[node_idx]
        else:
            pin_pos[pin_idx] = pin_offset[pin_idx]

def get_wl_xy(pin2net, pin2node, pin_offset, node_pos, net_cnt, node_cnt, gamma):
    threads_per_block = args.threads_per_block

    pin2net_gpu = cuda.to_device(pin2net)
    pin2node_gpu = cuda.to_device(pin2node)
    pin_cnt = pin2node.shape[0]
    all_cnt = pin2net.shape[0]

    pin_pos_gpu = cuda.device_array(shape=(all_cnt), dtype=np.float32)
    node_pos_gpu = cuda.to_device(node_pos)
    pin_offset_gpu = cuda.to_device(pin_offset)
    blocks_per_grid = min((pin_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_pin_pos[blocks_per_grid, threads_per_block](pin_pos_gpu, pin_offset_gpu, node_pos_gpu, pin2node_gpu, pin_cnt)

    e_plus_init = np.zeros((net_cnt), dtype=np.float32)
    e_plus_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    e_plus_gpu.copy_to_device(e_plus_init)
    e_minus_init = np.ones((net_cnt), dtype=np.float32)
    e_minus_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    e_minus_gpu.copy_to_device(e_minus_init)
    blocks_per_grid = min((pin_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_e_plus_minus[blocks_per_grid, threads_per_block](e_plus_gpu, e_minus_gpu, pin_pos_gpu, pin2net_gpu)

    a_plus_gpu = cuda.device_array(shape=(all_cnt), dtype=np.float32)
    a_minus_gpu = cuda.device_array(shape=(all_cnt), dtype=np.float32)
    blocks_per_grid = min((pin_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_a_plus_minus[blocks_per_grid, threads_per_block](a_plus_gpu, a_minus_gpu, e_plus_gpu, e_minus_gpu, pin_pos_gpu, pin2net_gpu, gamma)
    
    bc_init = np.zeros((net_cnt), dtype=np.float32)
    b_plus_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    b_plus_gpu.copy_to_device(bc_init)
    b_minus_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    b_minus_gpu.copy_to_device(bc_init)
    blocks_per_grid = min((all_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_b_plus_minus[blocks_per_grid, threads_per_block](b_plus_gpu, b_minus_gpu, a_plus_gpu, a_minus_gpu, pin2net_gpu)

    bc_init = np.zeros((net_cnt), dtype=np.float32)
    c_plus_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    c_plus_gpu.copy_to_device(bc_init)
    c_minus_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    c_minus_gpu.copy_to_device(bc_init)
    blocks_per_grid = min((all_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_c_plus_minus[blocks_per_grid, threads_per_block](c_plus_gpu, c_minus_gpu, a_plus_gpu, a_minus_gpu, pin_pos_gpu, pin2net_gpu)

    wl_gpu = cuda.device_array(shape=(1), dtype=np.float32)
    wl_gpu.copy_to_device(np.zeros((1), dtype=np.float32))
    blocks_per_grid = min((net_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_tol_wl[blocks_per_grid, threads_per_block](wl_gpu, b_plus_gpu, b_minus_gpu, c_plus_gpu, c_minus_gpu)

    dpos_init = np.zeros((node_cnt), dtype=np.float32)
    dpos_gpu = cuda.device_array(shape=(node_cnt), dtype=np.float32)
    dpos_gpu.copy_to_device(dpos_init)
    blocks_per_grid = min((all_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_dpos[blocks_per_grid, threads_per_block](dpos_gpu, 
             a_plus_gpu, a_minus_gpu, 
             b_plus_gpu, b_minus_gpu, 
             c_plus_gpu, c_minus_gpu, 
             pin_pos_gpu, pin2net_gpu,
             pin2node_gpu, gamma, pin_cnt)
    
    wl = wl_gpu.copy_to_host()
    dpos = dpos_gpu.copy_to_host()
    return wl[0], dpos.reshape(-1, 1)

@cuda.jit
def cal_net_bound(net_bound_min, net_bound_max, pin2net, pin_pos):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    for pin_idx in range(idx_start, pin_pos.shape[0], threads_per_grid):
        net_idx = pin2net[pin_idx]
        cuda.atomic.max(net_bound_max, net_idx, pin_pos[pin_idx])
        cuda.atomic.min(net_bound_min, net_idx, pin_pos[pin_idx])

def get_hpwl_xy(pin2net, pin2node, pin_offset, node_pos, net_cnt):
    threads_per_block = args.threads_per_block
    blocks_per_grid = args.blocks_per_grid

    pin2net_gpu = cuda.to_device(pin2net)
    pin2node_gpu = cuda.to_device(pin2node)
    pin_cnt = pin2node.shape[0]
    all_cnt = pin2net.shape[0]

    pin_pos_gpu = cuda.device_array(shape=(all_cnt), dtype=np.float32)
    node_pos_gpu = cuda.to_device(node_pos)
    pin_offset_gpu = cuda.to_device(pin_offset)
    blocks_per_grid = min((pin_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_pin_pos[blocks_per_grid, threads_per_block](pin_pos_gpu, pin_offset_gpu, node_pos_gpu, pin2node_gpu, pin_cnt)

    net_bound_min_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    net_bound_max_gpu = cuda.device_array(shape=(net_cnt), dtype=np.float32)
    net_bound_min_gpu.copy_to_device(np.ones((net_cnt), dtype=np.float32) * 1e10)
    net_bound_max_gpu.copy_to_device(np.ones((net_cnt), dtype=np.float32) * (-1e10))
    blocks_per_grid = min((pin_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    cal_net_bound[blocks_per_grid, threads_per_block](net_bound_min_gpu, net_bound_max_gpu, pin2net_gpu, pin_pos_gpu)
    net_bound_min = net_bound_min_gpu.copy_to_host()
    net_bound_max = net_bound_max_gpu.copy_to_host()
    hpwlx = np.sum(net_bound_max - net_bound_min)
    return hpwlx

def get_wl(chip : RefineDB, pos, ref_args):
    global args
    args = ref_args
    
    pin_offset = chip.pin_offset / chip.chip_size.reshape(-1, 2)
    pin2net = chip.pin2net
    pos_all = pos

    wlx, dposx = get_wl_xy(pin2net, chip.pin2node, np.ascontiguousarray(pin_offset[:, 0]), np.ascontiguousarray(pos_all[:, 0]), chip.net_cnt, chip.node_cnt, args.gamma)
    wly, dposy = get_wl_xy(pin2net, chip.pin2node, np.ascontiguousarray(pin_offset[:, 1]), np.ascontiguousarray(pos_all[:, 1]), chip.net_cnt, chip.node_cnt, args.gamma)
    return wlx + wly, np.concatenate([dposx, dposy], axis = 1)

def get_hpwl(chip : RefineDB, pos, ref_args):
    global args
    args = ref_args
    pin_offset = chip.pin_offset
    pin2net = chip.pin2net
    pos_all = pos
    hpwlx = get_hpwl_xy(pin2net, chip.pin2node, np.ascontiguousarray(pin_offset[:, 0]), np.ascontiguousarray(pos_all[:, 0]), chip.net_cnt)
    hpwly = get_hpwl_xy(pin2net, chip.pin2node, np.ascontiguousarray(pin_offset[:, 1]), np.ascontiguousarray(pos_all[:, 1]), chip.net_cnt)
    return hpwlx + hpwly
