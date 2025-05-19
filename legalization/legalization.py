import torch
import torch.nn as nn
import numpy as np
import sys
import os
import legalization.dreamplace.dreamplace.ops.legality_check.legality_check as legality_check
import legalization.dreamplace.dreamplace.ops.macro_legalize.macro_legalize as macro_legalize
import legalization.dreamplace.dreamplace.ops.greedy_legalize.greedy_legalize as greedy_legalize
import legalization.dreamplace.dreamplace.ops.abacus_legalize.abacus_legalize as abacus_legalize

class PlaceDataCollection():
    def __init__(self, refinedb, args, num_threads=4, device='cuda'):
        torch.set_num_threads(num_threads)
        self.xl, self.yl = 0, 0
        self.xh, self.yh = refinedb.chip_size[0], refinedb.chip_size[1]
        self.num_movable_nodes = refinedb.node_size.shape[0]
        self.benchmark = refinedb.benchmark
        self.site_width, self.row_height = self.read_scl(args)
        self.num_bins_x, self.num_bins_y = 2048, 2048
        node2fence_region_map = np.ones((self.num_movable_nodes), dtype=np.int32) * 2147483647
        num_pins_in_nodes = self.get_num_pins(refinedb)
        with torch.no_grad():
            self.node_size_x = torch.from_numpy(refinedb.node_size[:, 0]).to(device, dtype=torch.float32)
            self.node_size_y = torch.from_numpy(refinedb.node_size[:, 1]).to(device, dtype=torch.float32)
            self.flat_region_boxes = torch.from_numpy(np.array([], dtype=np.float32)).to(device)
            self.flat_region_boxes_start = torch.from_numpy(np.array([0], dtype=np.int32)).to(device)
            self.node2fence_region_map = torch.from_numpy(node2fence_region_map).to(device)
            self.num_pins_in_nodes = torch.from_numpy(num_pins_in_nodes).to(device)
        self.num_threads = num_threads
        self.device = device
    
    def get_num_pins(self, refinedb):
        num_pins_in_nodes = np.zeros((refinedb.node_size.shape[0]), dtype=np.float32)
        for node in refinedb.pin2node:
            num_pins_in_nodes[node] += 1
        return num_pins_in_nodes

    def read_scl(self, args):
        file = os.path.join(args.benchmark_dir, f'{self.benchmark}/{self.benchmark}.scl')
        with open(file, 'r') as f:
            line = f.readline()
            while line:
                line = line.strip()
                if line.startswith('Height'):
                    line_list = line.split()
                    row_height = float(line_list[2])
                if line.startswith('Sitewidth'):
                    line_list = line.split()
                    site_width = float(line_list[2])
                    break
                line = f.readline()
        return site_width, row_height
            

class MacroLegalization():
    def __init__(self, refinedb, args):
        self.data_collections = PlaceDataCollection(refinedb, args)
        self.legalize_op = self.build_legalization(self.data_collections)
        self.legality_check_op = self.build_legality_check(self.data_collections)
    
    def build_legalization(self, data_collections):
        ml = macro_legalize.MacroLegalize(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=data_collections.xl,
            yl=data_collections.yl,
            xh=data_collections.xh,
            yh=data_collections.yh,
            site_width=data_collections.site_width,
            row_height=data_collections.row_height,
            num_bins_x=data_collections.num_bins_x,
            num_bins_y=data_collections.num_bins_y,
            num_movable_nodes=data_collections.num_movable_nodes,
            num_terminal_NIs=0,
            num_filler_nodes=0)
        
        legalize_alg = greedy_legalize.GreedyLegalize
        gl = legalize_alg(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=data_collections.xl,
            yl=data_collections.yl,
            xh=data_collections.xh,
            yh=data_collections.yh,
            site_width=data_collections.site_width,
            row_height=data_collections.row_height,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=data_collections.num_movable_nodes,
            num_terminal_NIs=0,
            num_filler_nodes=0)

        al = abacus_legalize.AbacusLegalize(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            node_weights=data_collections.num_pins_in_nodes,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=data_collections.xl,
            yl=data_collections.yl,
            xh=data_collections.xh,
            yh=data_collections.yh,
            site_width=data_collections.site_width,
            row_height=data_collections.row_height,
            num_bins_x=1,
            num_bins_y=64,
            num_movable_nodes=data_collections.num_movable_nodes,
            num_terminal_NIs=0,
            num_filler_nodes=0)
        
        def build_legalization_op(pos):
            print("Start legalization")
            pos1 = ml(pos, pos)
            pos2 = gl(pos1, pos1)

            legal = self.legality_check_op(pos2)
            if not legal:
                print("legality check failed in greedy legalization, " \
                    "return illegal results after greedy legalization.")
            pos3 = al(pos1, pos2)
            legal = self.legality_check_op(pos3)
            if not legal:
                print("legality check failed in abacus legalization, " \
                    "return legal results after greedy legalization.")
            else:
                print('-----legalization success-----')
            return pos3, legal
  
        return build_legalization_op

    def build_legality_check(self, data_collections):
        return legality_check.LegalityCheck(
            node_size_x=data_collections.node_size_x,
            node_size_y=data_collections.node_size_y,
            flat_region_boxes=data_collections.flat_region_boxes,
            flat_region_boxes_start=data_collections.flat_region_boxes_start,
            node2fence_region_map=data_collections.node2fence_region_map,
            xl=data_collections.xl,
            yl=data_collections.yl,
            xh=data_collections.xh,
            yh=data_collections.yh,
            site_width=data_collections.site_width,
            row_height=data_collections.row_height,
            scale_factor=1.0,
            num_terminals=0,
            num_movable_nodes=data_collections.num_movable_nodes)

    def legalize(self, pos):
        init_pos = np.zeros(self.data_collections.num_movable_nodes * 2, dtype=np.float32)
        init_pos[:self.data_collections.num_movable_nodes] = pos[:, 0]
        init_pos[self.data_collections.num_movable_nodes:] = pos[:, 1]
        pos_tensor = nn.ParameterList([nn.Parameter(torch.from_numpy(init_pos).to(self.data_collections.device))])
        legal_pos = self.legalize_op(pos_tensor[0])
        return legal_pos