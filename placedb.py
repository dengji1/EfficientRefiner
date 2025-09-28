import numpy as np
import os
from itertools import combinations

class PlaceDB:
    def __init__(self, benchmark, base_path, pl_path, args):
        self.benchmark = benchmark
        self.node_info, self.net_info, self.pin_info = {}, {}, {}
        self.node_info_all = {}
        self.expert_pos = {}
        self.width, self.height = 0, 0
        self.offset_x, self.offset_y = 0, 0
        self.openroad = False
        if 'openroad' in base_path:
            self.openroad = True
        
        if not base_path.endswith(args.macro_suffix):
            base_path_full = base_path
            base_path += args.macro_suffix
        else:
            base_path_full = base_path[:-len(args.macro_suffix)]
        base_path_full = os.path.join(base_path_full, benchmark)
        
        self.get_all_nodes(base_path_full)
        self.get_orign(base_path_full)
            
        base_path = os.path.join(base_path, benchmark)
        self.get_node_info(base_path)
        self.get_net_pin_info(base_path)
        if self.openroad:
            self.get_size_from_pl(base_path_full)
        else:
            self.get_size()
        self.get_expert_pos(pl_path)
    
    def get_orign(self, route):
        path = os.path.join(route, self.benchmark+'.pl')
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) <= 4 or parts[0] == '#':
                    continue
                node = parts[0]
                orign = parts[4]
                self.node_info_all[node]['orign'] = orign
    
    def get_all_nodes(self, route):
        path = os.path.join(route, self.benchmark+'.nodes')
        with open(path, 'r') as f:
            for line in f:
                node_line = line.split()
                if not node_line:
                    continue
                if line.startswith('\t') or line.startswith(' '):
                    node_line = line.split()
                    self.node_info_all[node_line[0]] = {}
                    self.node_info_all[node_line[0]]['x'] = float(node_line[1])
                    self.node_info_all[node_line[0]]['y'] = float(node_line[2])
                    self.node_info_all[node_line[0]]['port'] = False
                    if len(node_line) > 3 and node_line[3] == 'terminal_NI':
                        self.node_info_all[node_line[0]]['port'] = True

    def get_size_from_pl(self, route):
        pl_path = os.path.join(route, self.benchmark+'.ref.pl')
        xl, yl, xh, yh = float('inf'), float('inf'), float('-inf'), float('-inf')
        with open(pl_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) > 4:
                    node = parts[0]
                    x, y = float(parts[1]), float(parts[2])
                    xl = min(x, xl)
                    yl = min(y, yl)
                    xh = max(x + self.node_info_all[node]['x'], xh)
                    yh = max(y + self.node_info_all[node]['y'], yh)
        self.offset_x, self.offset_y = xl, yl
        self.width = xh - xl
        self.height = yh - yl
    
    def get_node_info(self, route):
        node_id = 0
        path = os.path.join(route, self.benchmark+'.nodes')
        with open(path, 'r') as f:
            for line in f:
                node_line = line.split()
                if not node_line:
                    continue
                if line.startswith('\t') or line.startswith(' '):
                    node_line = line.split()
                    self.node_info[node_line[0]] = {}
                    self.node_info[node_line[0]]['x'] = float(node_line[1])
                    self.node_info[node_line[0]]['y'] = float(node_line[2])
                    self.node_info[node_line[0]]['id'] = node_id
                    self.node_info[node_line[0]]['fix'] = False
                    self.node_info[node_line[0]]['pins'] = []
                    if node_line[-1]=='terminal_NI':
                        self.node_info[node_line[0]]['fix'] = True
                    node_id += 1
    
    def get_net_pin_info(self, route):
        net_id, pin_id = 0, 0
        path = os.path.join(route, self.benchmark+'.nets')
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('NetDegree'):
                    net_line = line.split()
                    self.net_info[net_id] = {}
                    self.net_info[net_id]['net_name'] = net_line[3]
                    self.net_info[net_id]['pins'] = []
                    nodes = set()
                    for _ in range(int(net_line[2])):
                        line = f.readline()
                        pin_line = line.split()
                        if pin_line[0] not in nodes and pin_line[0] in self.node_info:
                            nodes.add(pin_line[0])
                            self.pin_info[pin_id] = {}
                            self.pin_info[pin_id]['node'] = pin_line[0]
                            self.pin_info[pin_id]['net'] =net_id
                            self.pin_info[pin_id]['x'] = float(pin_line[-2])
                            self.pin_info[pin_id]['y'] = float(pin_line[-1])
                            self.net_info[net_id]['pins'].append(pin_id)
                            self.node_info[pin_line[0]]['pins'].append(pin_id)
                            pin_id += 1
                        
                    if len(self.net_info[net_id]['pins']) < 2:
                        for pin_id1 in self.net_info[net_id]['pins']:
                            node = self.pin_info[pin_id1]['node']
                            self.node_info[node]['pins'] = self.node_info[node]['pins'][:-1]
                            del self.pin_info[pin_id1]
                            pin_id -= 1
                        del self.net_info[net_id]
                    else:
                        net_id += 1
    
    def get_size(self):
        if (self.benchmark) == 'adaptec1':
            self.width = 11589
            self.height = 11589
        if (self.benchmark) == 'adaptec2':
            self.width = 15244
            self.height = 15244
        if (self.benchmark) == 'adaptec3':
            self.width = 23386
            self.height = 23386
        if (self.benchmark) == 'adaptec4':
            self.width = 23386
            self.height = 23386
        if (self.benchmark) == 'bigblue1':
            self.width = 11589
            self.height = 11589
        if (self.benchmark) == 'bigblue2':
            self.width = 23084
            self.height = 23084
        if (self.benchmark) == 'bigblue3':
            self.width = 27868
            self.height = 27868
        if (self.benchmark) == 'bigblue4':
            self.width = 32386
            self.height = 32386
        if (self.benchmark) == 'superblue1':
            self.width = 17086320
            self.height = 6262020
        if (self.benchmark) == 'superblue3':
            self.width = 19419520
            self.height = 6299640
        if (self.benchmark) == 'superblue4':
            self.width = 11310320
            self.height = 6299640
        if (self.benchmark) == 'superblue5':
            self.width = 18797080
            self.height = 8649180
        if (self.benchmark) == 'superblue7':
            self.width = 12093120
            self.height = 10824300
        if (self.benchmark) == 'superblue10':
            self.width = 16526960
            self.height = 11699820
        if (self.benchmark) == 'superblue16':
            self.width = 11274600
            self.height = 6121800
        if (self.benchmark) == 'superblue18':
            self.width = 9227160
            self.height = 6101280
    
    def get_expert_pos(self, path):
        with open(path,'r') as f:
            for line in f:
                pos_line = line.split()
                if len(pos_line) >= 4 and not line.startswith('#'):
                    if pos_line[0] in self.node_info:
                        self.expert_pos[pos_line[0]] = {}
                        self.expert_pos[pos_line[0]]['x'] = float(pos_line[1]) - self.offset_x
                        self.expert_pos[pos_line[0]]['y'] = float(pos_line[2]) - self.offset_y
                        self.expert_pos[pos_line[0]]['orign'] = pos_line[4] 

class RefineDB():
    def __init__(self, placedb: PlaceDB):
        self.benchmark = placedb.benchmark
        self.id2name_node, self.name2id_node, self.raw_node_pos = [], {}, []
        self.id2name_net, self.name2id_net = [], {}
        self.pin_offset, self.pin2node, self.pin2net = [], [], []
        self.node_size = np.zeros((len(placedb.node_info), 2))
        self.node_info = {}
        self.fix_mask = np.zeros(len(placedb.node_info), dtype=np.bool_)
        
        self.get_node_info(placedb)
        self.get_net_info(placedb)
        self.chip_size = np.array([placedb.width, placedb.height])
        self.get_node_pos(placedb)
        
        self.node_cnt = len(self.id2name_node)
        self.pin_cnt = len(self.pin2node)
        self.net_cnt = len(self.id2name_net)

    def get_node_info(self, placedb: PlaceDB):
        cnt = 0
        for node in placedb.node_info:
            if not placedb.node_info[node]['fix']:
                self.id2name_node.append(node)
                self.name2id_node[node] = cnt
                x, y = placedb.node_info[node]['x'], placedb.node_info[node]['y']
                self.node_info[node] = {'x': x, 'y': y}
                self.node_size[cnt, 0], self.node_size[cnt, 1] = x, y
                self.fix_mask[cnt] = False
                cnt += 1
        for node in placedb.node_info:
            if placedb.node_info[node]['fix']:
                self.id2name_node.append(node)
                self.name2id_node[node] = cnt
                x, y = placedb.node_info[node]['x'], placedb.node_info[node]['y']
                self.node_info[node] = {'x': x, 'y': y}
                self.node_size[cnt, 0], self.node_size[cnt, 1] = x, y
                self.fix_mask[cnt] = True
                cnt += 1
    
    def get_node_pos(self, placedb:PlaceDB):
        for node in self.id2name_node:
            self.raw_node_pos.append([placedb.expert_pos[node]['x'], placedb.expert_pos[node]['y']])
        self.raw_node_pos = np.array(self.raw_node_pos)
    
    def get_net_info(self, placedb: PlaceDB):
        net_cnt = 0
        for net in placedb.net_info:
            if len(placedb.net_info[net]['pins']) < 2:
                continue
            self.id2name_net.append(net)
            self.name2id_net[net] = net_cnt
            for pin in placedb.net_info[net]['pins']:
                node = placedb.pin_info[pin]['node']
                x_offset, y_offset = placedb.pin_info[pin]['x'], placedb.pin_info[pin]['y']
                self.pin_offset.append([x_offset, y_offset])
                self.pin2node.append(self.name2id_node[node])
                self.pin2net.append(self.name2id_net[net])
            net_cnt += 1
            
        self.pin_offset = np.array(self.pin_offset)
        self.pin2node = np.array(self.pin2node, dtype=np.int32)
        self.pin2net = np.array(self.pin2net, dtype=np.int32)
