import numpy as np
import os

class PlaceDB:
    def __init__(self, benchmark, base_path, pl_path):
        self.benchmark = benchmark
        self.node_info, self.net_info, self.pin_info, self.port_info = {}, {}, {}, {}
        self.expert_pos = {}
        self.width, self.height = 0, 0
        
        self.get_size()
        base_path = os.path.join(base_path, benchmark)
        self.get_node_info(base_path)
        self.get_net_pin_info(base_path)
        self.get_expert_pos(pl_path)

    def get_node_info(self, route):
        node_id = 0
        path = os.path.join(route, self.benchmark+'.nodes')
        with open(path, 'r') as f:
            for line in f:
                node_line = line.split()
                if not node_line:
                    continue
                if node_line[0].startswith('p'):
                        self.port_info[node_line[0]] = {}
                        self.port_info[node_line[0]]['size_x'] = int(node_line[1])
                        self.port_info[node_line[0]]['size_y'] = int(node_line[2])
                        self.port_info[node_line[0]]['nets'] = []
                else:
                    if line.startswith('\t') or line.startswith(' '):
                        node_line = line.split()
                        self.node_info[node_line[0]] = {}
                        self.node_info[node_line[0]]['x'] = int(node_line[1])
                        self.node_info[node_line[0]]['y'] = int(node_line[2])
                        self.node_info[node_line[0]]['id'] = node_id
                        self.node_info[node_line[0]]['pins'] = []
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
                    self.net_info[net_id]['ports'] = []
                    nodes = set()
                    for _ in range(int(net_line[2])):
                        line = f.readline()
                        pin_line = line.split()
                        if pin_line[0].startswith('p'):
                            self.port_info[pin_line[0]]['nets'].append(net_id)
                            self.net_info[net_id]['ports'].append(pin_line[0])
                        if pin_line[0] not in nodes:
                            nodes.add(pin_line[0])
                            self.pin_info[pin_id] = {}
                            self.pin_info[pin_id]['node'] = pin_line[0]
                            self.pin_info[pin_id]['net'] =net_id
                            self.pin_info[pin_id]['x'] = float(pin_line[-2])
                            self.pin_info[pin_id]['y'] = float(pin_line[-1])
                            self.net_info[net_id]['pins'].append(pin_id)
                            self.node_info[pin_line[0]]['pins'].append(pin_id)
                        pin_id += 1
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
    
    def get_expert_pos(self, route):
        path = os.path.join(route, self.benchmark+'.pl')
        with open(path,'r') as f:
            for line in f:
                pos_line = line.split()
                if len(pos_line) >= 4 and not line.startswith('#'):
                    if pos_line[0].startswith('p'):
                        self.port_info[pos_line[0]]['x'] = float(pos_line[1])
                        self.port_info[pos_line[0]]['y'] = float(pos_line[2])
                    else:
                        self.expert_pos[pos_line[0]] = {}
                        self.expert_pos[pos_line[0]]['x'] = float(pos_line[1])
                        self.expert_pos[pos_line[0]]['y'] = float(pos_line[2])

class RefineDB():
    def __init__(self, placedb: PlaceDB):
        self.benchmark = placedb.benchmark
        self.id2name_node, self.name2id_node, self.raw_node_pos = [], {}, []
        self.id2name_net, self.name2id_net = [], {}
        self.pin_offset, self.pin2node, self.pin2net = [], [], []
        self.port_pos, self.port2net = [], []
        self.node_size = np.zeros((len(placedb.node_info), 2))
        self.node_info = {}
        
        self.get_node_info(placedb)
        self.get_net_info(placedb)
        self.chip_size = np.array([placedb.width, placedb.height])
        self.get_node_pos(placedb)
        
        self.node_cnt = len(self.id2name_node)
        self.pin_cnt = len(self.pin2node)
        self.net_cnt = len(self.id2name_net)
        self.port_cnt = len(self.port2net)

    def get_node_info(self, placedb: PlaceDB):
        cnt = 0
        for node in placedb.node_info:
            self.id2name_node.append(node)
            self.name2id_node[node] = cnt
            x, y = placedb.node_info[node]['x'], placedb.node_info[node]['y']
            self.node_info[node] = {'x': x, 'y': y}
            self.node_size[cnt, 0], self.node_size[cnt, 1] = x, y
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
            for port in placedb.net_info[net]['ports']:
                self.port_pos.append([placedb.port_info[port]['x'], placedb.port_info[port]['y']])
                self.port2net.append(self.name2id_net[net])
            net_cnt += 1
            
        self.pin_offset = np.array(self.pin_offset)
        self.pin2node = np.array(self.pin2node, dtype=np.int32)
        self.pin2net = np.array(self.pin2net, dtype=np.int32)
        self.port_pos = np.array(self.port_pos)
        self.port2net = np.array(self.port2net, dtype=np.int32)