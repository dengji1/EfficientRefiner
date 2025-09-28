from placedb import RefineDB
import numpy as np
from numba import cuda
import numba

@cuda.jit
def cal_o_dp(o, dp, pos, size, fix):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)

    node_cnt = pos.shape[0]
    for idx1 in range(ix, node_cnt, threads_per_grid_x):
        for idx2 in range(iy, node_cnt, threads_per_grid_y):
            if fix[idx1] == 1 and fix[idx2] == 1:
                o[idx1, idx2] = 0
                dp[idx1, idx2] = 0
            elif idx1 != idx2 and -(size[idx1] + size[idx2]) / 2 < pos[idx1] - pos[idx2] < (size[idx1] + size[idx2]) / 2:
                if pos[idx2] < pos[idx1]:
                    o[idx1, idx2] = (pos[idx2] + size[idx2] / 2) - (pos[idx1] - size[idx1] / 2)
                    dp[idx1, idx2] = -1
                else:
                    o[idx1, idx2] = (pos[idx1] + size[idx1] / 2) - (pos[idx2] - size[idx2] / 2)
                    dp[idx1, idx2] = 1
            else:
                o[idx1, idx2] = 0
                dp[idx1, idx2] = 0

@cuda.jit
def cal_overlap(overlap, ox, oy, fix):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)

    node_cnt = ox.shape[0]
    for idx1 in range(ix, node_cnt, threads_per_grid_x):
        for idx2 in range(iy, node_cnt, threads_per_grid_y):
            if idx1 < idx2 and not(fix[idx1] == 1 and fix[idx2] == 1):
                cuda.atomic.add(overlap, 0, ox[idx1, idx2] * oy[idx1, idx2])

@cuda.jit
def cal_dpos(dpos, o, dp, fix):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)
    node_cnt = o.shape[0]

    for idx1 in range(ix, node_cnt, threads_per_grid_x):
        for idx2 in range(iy, node_cnt, threads_per_grid_y):
            if not(fix[idx1] == 1 and fix[idx2] == 1):
                cuda.atomic.add(dpos, idx1, o[idx1, idx2] * dp[idx1, idx2])


def get_overlap(chip : RefineDB, pos, args, fix_mask):
    threads_per_block_2d = args.threads_per_block_2d
    blocks_per_grid_2d = args.blocks_per_grid_2d

    fix_gpu = cuda.to_device(np.ascontiguousarray(fix_mask))
    
    pos_x_gpu = cuda.to_device(np.ascontiguousarray(pos[:, 0]))
    size_x_gpu = cuda.to_device(np.ascontiguousarray(chip.node_size[:, 0] / chip.chip_size[0]))
    ox_gpu = cuda.device_array(shape=(chip.node_cnt, chip.node_cnt), dtype=np.float32)
    dpx_gpu = cuda.device_array(shape=(chip.node_cnt, chip.node_cnt), dtype=np.float32)
    cal_o_dp[blocks_per_grid_2d, threads_per_block_2d](ox_gpu, dpx_gpu, pos_x_gpu, size_x_gpu, fix_gpu)
    
    pos_y_gpu = cuda.to_device(np.ascontiguousarray(pos[:, 1]))
    size_y_gpu = cuda.to_device(chip.node_size[:, 1] / chip.chip_size[1])
    oy_gpu = cuda.device_array(shape=(chip.node_cnt, chip.node_cnt), dtype=np.float32)
    dpy_gpu = cuda.device_array(shape=(chip.node_cnt, chip.node_cnt), dtype=np.float32)
    cal_o_dp[blocks_per_grid_2d, threads_per_block_2d](oy_gpu, dpy_gpu, pos_y_gpu, size_y_gpu, fix_gpu)

    overlap_gpu = cuda.device_array(shape=(1), dtype=np.float32)
    overlap_gpu.copy_to_device(np.zeros((1), dtype=np.float32))
    cal_overlap[blocks_per_grid_2d, threads_per_block_2d](overlap_gpu, ox_gpu, oy_gpu, fix_gpu)

    dposx_gpu = cuda.device_array(shape=(chip.node_cnt), dtype=np.float32)
    dposx_gpu.copy_to_device(np.zeros((chip.node_cnt), dtype=np.float32))
    cal_dpos[blocks_per_grid_2d, threads_per_block_2d](dposx_gpu, oy_gpu, dpx_gpu, fix_gpu)

    dposy_gpu = cuda.device_array(shape=(chip.node_cnt), dtype=np.float32)
    dposy_gpu.copy_to_device(np.zeros((chip.node_cnt), dtype=np.float32))
    cal_dpos[blocks_per_grid_2d, threads_per_block_2d](dposy_gpu, ox_gpu, dpy_gpu, fix_gpu)

    overlap = overlap_gpu.copy_to_host()
    dposx, dposy = dposx_gpu.copy_to_host(), dposy_gpu.copy_to_host()

    return overlap[0], np.concatenate([dposx.reshape(-1,1), dposy.reshape(-1,1)], axis=1)

@cuda.jit
def cal_eo_xy(ox, pos_x, size_x):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)
    node_cnt = pos_x.shape[0]

    for idx1 in range(ix, node_cnt, threads_per_grid_x):
        for idx2 in range(iy, node_cnt, threads_per_grid_y):
            if idx1 != idx2:
                ox[idx1, idx2] = max(min(pos_x[idx1] + size_x[idx1] / 2, pos_x[idx2] + size_x[idx2] / 2) - max(pos_x[idx1] - size_x[idx1] / 2, pos_x[idx2] - size_x[idx2] / 2), 0)
            else:
                ox[idx1, idx2] = 0

def get_exact_overlap(chip : RefineDB, pos, args):
    threads_per_block_2d = args.threads_per_block_2d
    blocks_per_grid_2d = args.blocks_per_grid_2d

    pos_x_gpu = cuda.to_device(np.ascontiguousarray(pos[:, 0]))
    size_x_gpu = cuda.to_device(np.ascontiguousarray(chip.node_size[:, 0] / chip.chip_size[0]))
    ox_gpu = cuda.device_array(shape=(chip.node_cnt, chip.node_cnt), dtype=np.float32)
    cal_eo_xy[blocks_per_grid_2d, threads_per_block_2d](ox_gpu, pos_x_gpu, size_x_gpu)

    pos_y_gpu = cuda.to_device(np.ascontiguousarray(pos[:, 1]))
    size_y_gpu = cuda.to_device(np.ascontiguousarray(chip.node_size[:, 1]/ chip.chip_size[1]))
    oy_gpu = cuda.device_array(shape=(chip.node_cnt, chip.node_cnt), dtype=np.float32)
    cal_eo_xy[blocks_per_grid_2d, threads_per_block_2d](oy_gpu, pos_y_gpu, size_y_gpu)

    ox, oy = ox_gpu.copy_to_host(), oy_gpu.copy_to_host()
    eo = ox * oy
    return eo.sum() / 2 * 100
