from placedb import RefineDB
import numpy as np
from numba import cuda
import numba

@cuda.jit(device=True)
def cal_pair_ol_dpos(sx1, sx2, sy1, sy2, px1, px2, py1, py2):
    if -(sx1 + sx2) / 2 < px1 - px2 < (sx1 + sx2) / 2:
        if px2 < px1:
            ox = (px2 + sx2 / 2) - (px1 - sx1 / 2)
            dpx = -1
        else:
            ox = (px1 + sx1 / 2) - (px2 - sx2 / 2)
            dpx = 1
    else:
        ox = 0
        dpx = 0
        
    if -(sy1 + sy2) / 2 < py1 - py2 < (sy1 + sy2) / 2:
        if py2 < py1:
            oy = (py2 + sy2 / 2) - (py1 - sy1 / 2)
            dpy = -1
        else:
            oy = (py1 + sy1 / 2) - (py2 - sy2 / 2)
            dpy = 1
    else:
        oy = 0
        dpy = 0
    return ox, oy, dpx, dpy

@cuda.jit
def cal_overlap_dpos(overlap, dposx, dposy, num_bins,
                     bin_module_cnt, bin_modules, bin_x, bin_y, 
                     pos_x, pos_y, size_x, size_y, big_modules):
    
    ix, iy = cuda.grid(2) 
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2) 

    node_cnt = pos_x.shape[0]
    for m_idx1 in range(ix, node_cnt, threads_per_grid_x): 
        if big_modules[m_idx1]:
            for m_idx2 in range(iy, node_cnt, threads_per_grid_y):
                if (big_modules[m_idx2] and m_idx1 < m_idx2) or (not big_modules[m_idx2]):
                    ox, oy, dpx, dpy = cal_pair_ol_dpos(
                    size_x[m_idx1], size_x[m_idx2], size_y[m_idx1], size_y[m_idx2],
                    pos_x[m_idx1], pos_x[m_idx2], pos_y[m_idx1], pos_y[m_idx2])                              
                    if ox > 0 and oy > 0:
                        cuda.atomic.add(overlap, 0, ox * oy)
                        cuda.atomic.add(dposx, m_idx1, oy * dpx)
                        cuda.atomic.add(dposx, m_idx2, -oy * dpx)
                        cuda.atomic.add(dposy, m_idx1, ox * dpy)
                        cuda.atomic.add(dposy, m_idx2, -ox * dpy)
        else:
            max_nei = 9 * bin_modules.shape[1]
            bx = int(pos_x[m_idx1] / bin_x)
            by = int(pos_y[m_idx1] / bin_y)
            for idx in range(iy, max_nei, threads_per_grid_y):
                nei_bin_idx = idx // bin_modules.shape[1]
                nei_m_idx = idx % bin_modules.shape[1]
                off_x = nei_bin_idx // 3 - 1
                off_y = nei_bin_idx % 3 - 1
                if bx + off_x < 0 or bx + off_x >= num_bins:
                    continue
                if by + off_y < 0 or by + off_y >= num_bins:
                    continue
                bin_idx = (bx + off_x) * num_bins + (by + off_y)
                if nei_m_idx >= bin_module_cnt[bin_idx]:
                    continue
                m_idx2 = bin_modules[bin_idx][nei_m_idx]
                if m_idx1 < m_idx2:
                    ox, oy, dpx, dpy = cal_pair_ol_dpos(
                    size_x[m_idx1], size_x[m_idx2], size_y[m_idx1], size_y[m_idx2],
                    pos_x[m_idx1], pos_x[m_idx2], pos_y[m_idx1], pos_y[m_idx2])                              
                    if ox > 0 and oy > 0:
                        cuda.atomic.add(overlap, 0, ox * oy)
                        cuda.atomic.add(dposx, m_idx1, oy * dpx)
                        cuda.atomic.add(dposx, m_idx2, -oy * dpx)
                        cuda.atomic.add(dposy, m_idx1, ox * dpy)
                        cuda.atomic.add(dposy, m_idx2, -ox * dpy)

@cuda.jit
def get_bin_modules(bin_module_cnt, bin_modules, num_bins,   
    bin_x, bin_y, pos_x, pos_y, big_modules):
    
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    
    for m_idx in range(idx_start, pos_x.shape[0], threads_per_grid):
        if big_modules[m_idx]:
            continue
        bx = int(pos_x[m_idx] / bin_x)
        by = int(pos_y[m_idx] / bin_y)
        bin_idx = bx * num_bins + by
        cnt = cuda.atomic.add(bin_module_cnt, bin_idx, 1)
        if cnt < bin_modules.shape[1]:
            bin_modules[bin_idx, cnt] = m_idx 

@cuda.jit
def get_big_modules_cuda(big_modules, 
    bin_x, bin_y, size_x, size_y):
    idx_start = cuda.grid(1)
    threads_per_grid = cuda.blockDim.x * cuda.gridDim.x
    
    for m_idx in range(idx_start, size_x.shape[0], threads_per_grid):
        if bin_x < size_x[m_idx] or bin_y < size_y[m_idx]:
            big_modules[m_idx] = 1
        else:
            big_modules[m_idx] = 0

def get_big_modules(chip : RefineDB, num_bins, args):
    bin_x, bin_y = 1 / num_bins, 1 / num_bins
    threads_per_block = args.threads_per_block
    blocks_per_grid = min((chip.node_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    
    size_x_gpu = cuda.to_device(np.ascontiguousarray(chip.node_size[:, 0] / chip.chip_size[0]))
    size_y_gpu = cuda.to_device(np.ascontiguousarray(chip.node_size[:, 1] / chip.chip_size[1]))
    
    big_modules_gpu = cuda.device_array(shape=(chip.node_cnt), dtype=np.bool_)
    get_big_modules_cuda[blocks_per_grid, threads_per_block](big_modules_gpu, 
    bin_x, bin_y, size_x_gpu, size_y_gpu)
    big_modules = big_modules_gpu.copy_to_host()
    
    return big_modules
    
def get_overlap(chip : RefineDB, pos, big_modules, num_bins, args):
    threads_per_block_2d = args.threads_per_block_2d
    threads_per_block = args.threads_per_block
    module_per_bin = args.max_module_per_bin
    
    bin_x, bin_y = 1 / num_bins, 1 / num_bins
    size_x_gpu = cuda.to_device(np.ascontiguousarray(chip.node_size[:, 0] / chip.chip_size[0]))
    size_y_gpu = cuda.to_device(np.ascontiguousarray(chip.node_size[:, 1] / chip.chip_size[1]))
    big_modules_gpu = cuda.to_device(big_modules)
    
    bin_modules = cuda.device_array(shape=(num_bins * num_bins, module_per_bin), dtype=np.int32)
    bin_module_cnt = cuda.to_device(np.zeros((num_bins * num_bins), dtype=np.int32))
    pos_x_gpu = cuda.to_device(np.ascontiguousarray(pos[:, 0]))
    pos_y_gpu = cuda.to_device(np.ascontiguousarray(pos[:, 1]))
    blocks_per_grid = min((chip.node_cnt + threads_per_block - 1) // threads_per_block, args.blocks_per_grid)
    get_bin_modules[blocks_per_grid, threads_per_block](
    bin_module_cnt, bin_modules, num_bins, bin_x, bin_y, 
    pos_x_gpu, pos_y_gpu, big_modules_gpu) 
    
    overlap_gpu = cuda.device_array(shape=(1), dtype=np.float32)
    overlap_gpu.copy_to_device(np.zeros((1), dtype=np.float32))
    dposy_gpu = cuda.device_array(shape=(chip.node_cnt), dtype=np.float32)
    dposy_gpu.copy_to_device(np.zeros((chip.node_cnt), dtype=np.float32))
    dposx_gpu = cuda.device_array(shape=(chip.node_cnt), dtype=np.float32)
    dposx_gpu.copy_to_device(np.zeros((chip.node_cnt), dtype=np.float32))
    
    num_blocks = min((chip.node_cnt + threads_per_block_2d[0] - 1) // threads_per_block_2d[0], args.blocks_per_grid_2d[0])
    blocks_per_grid_2d = (num_blocks, num_blocks)
    cal_overlap_dpos[blocks_per_grid_2d, threads_per_block_2d](
       overlap_gpu, dposx_gpu, dposy_gpu, num_bins,
       bin_module_cnt, bin_modules, bin_x, bin_y, 
       pos_x_gpu, pos_y_gpu, size_x_gpu, size_y_gpu, big_modules_gpu
    )

    overlap = overlap_gpu.copy_to_host()
    dposx, dposy = dposx_gpu.copy_to_host(), dposy_gpu.copy_to_host()

    return overlap[0], np.concatenate([dposx.reshape(-1,1), dposy.reshape(-1,1)], axis=1)

@cuda.jit
def cal_eo_xy(ox, pos_x, size_x):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)
    node_cnt = pos_x.shape[0]

    for m_idx1 in range(ix, node_cnt, threads_per_grid_x):
        for m_idx2 in range(iy, node_cnt, threads_per_grid_y):
            if m_idx1 != m_idx2:
                ox[m_idx1, m_idx2] = max(min(pos_x[m_idx1] + size_x[m_idx1] / 2, pos_x[m_idx2] + size_x[m_idx2] / 2) - max(pos_x[m_idx1] - size_x[m_idx1] / 2, pos_x[m_idx2] - size_x[m_idx2] / 2), 0)
            else:
                ox[m_idx1, m_idx2] = 0


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