import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class AOP_NN(nn.Module):

    def __init__(self, dG, input_dim, num_hiddens_ratio):

        super(AOP_NN, self).__init__()
        self.num_hiddens_ratio = num_hiddens_ratio
        self.cal_term_dim(dG)
        self.input_dim = input_dim
        self.MIE_layer = self.construct_MIE_layer(dG)
        self.term_layer_list = self.construct_NN_graph(dG)

    def cal_term_dim(self, dG):
        self.term_dim_map = {}
        for term in dG.nodes():
            term_degree = dG.degree(term)
            num_output = term_degree * self.num_hiddens_ratio
            num_output = int(num_output)
            self.term_dim_map[term] = num_output

    def construct_MIE_layer(self, dG):
        MIEs = [n for n in dG.nodes() if dG.in_degree(n) == 0]
        self.MIE_layer = MIEs
        # for term in self.MIE_layer:
        #     self.add_module(term + '_input_layer', nn.Linear(self.input_dim, self.term_dim_map[term]))
        return self.MIE_layer

    def construct_NN_graph(self, dG):

        self.term_layer_list = []
        self.term_neighbor_map = {}
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.predecessors(term):
                self.term_neighbor_map[term].append(child)
        i = 0
        while True:
            leaves = [n for n in dG.nodes() if dG.in_degree(n) == 0]
            # print(i, 'len', len(leaves))
            i += 1
            if len(leaves) == 0:
                break

            for term in leaves:
                input_size = 0
                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                term_hidden = self.term_dim_map[term]

                if input_size != 0:
                    self.add_module(term + '_linear_layer', nn.Linear(input_size, term_hidden))
                    self.add_module(term + '_batchnorm_layer1', nn.BatchNorm1d(term_hidden))
                    self.add_module(term + '_aux_linear_layer', nn.Linear(term_hidden, 2))
                    self.add_module(term + '_batchnorm_layer2', nn.BatchNorm1d(2))
                else:
                    self.add_module(term + '_linear_layer', nn.Linear(self.input_dim, term_hidden))
                    self.add_module(term + '_batchnorm_layer1', nn.BatchNorm1d(term_hidden))
                    self.add_module(term + '_aux_linear_layer', nn.Linear(term_hidden, 2))
                    self.add_module(term + '_batchnorm_layer2', nn.BatchNorm1d(2))

            self.term_layer_list.append(leaves)
            dG.remove_nodes_from(leaves)

        return self.term_layer_list

    def forward(self, x):

        # MIE_input_out_map = {}
        # for term in self.MIE_layer:
        #     MIE_input_out_map[term] = self._modules[term+'_input_layer'](x)

        term_NN_out_map = {}
        aux_out_map = {}
        for i, layer in enumerate(self.term_layer_list):
            for term in layer:
                if term in self.MIE_layer:
                    child_input = x
                else:
                    child_input_list = []
                    for child in self.term_neighbor_map[term]:
                        child_input_list.append(term_NN_out_map[child])
                    child_input = torch.cat(child_input_list, 1)

                term_NN_out = torch.tanh(self._modules[term+'_linear_layer'](child_input))
                term_NN_out_map[term] = self._modules[term + '_batchnorm_layer1'](term_NN_out)
                aux_out = torch.tanh(self._modules[term + '_aux_linear_layer'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term + '_batchnorm_layer2'](aux_out)

        return aux_out_map, term_NN_out_map


class AOP_NN_feature(nn.Module):

    def __init__(self, dG, input_dim, num_hiddens_ratio):

        super(AOP_NN_feature, self).__init__()

        self.num_hiddens_ratio = num_hiddens_ratio
        self.cal_term_dim(dG)
        self.input_dim = input_dim
        self.MIE_layer = self.construct_MIE_layer(dG)
        self.term_layer_list = self.construct_NN_graph(dG)

    def cal_term_dim(self, dG):
        self.term_dim_map = {}
        for term in dG.nodes():
            term_degree = dG.degree(term)
            num_output = term_degree * self.num_hiddens_ratio
            num_output = int(num_output)
            self.term_dim_map[term] = num_output

    def construct_MIE_layer(self, dG):
        MIEs = [n for n in dG.nodes() if dG.in_degree(n) == 0]
        self.MIE_layer = MIEs
        # for term in self.MIE_layer:
        #     self.add_module(term + '_input_layer', nn.Linear(self.input_dim, self.term_dim_map[term]))
        return self.MIE_layer

    def construct_NN_graph(self, dG):

        self.term_layer_list = []
        self.term_neighbor_map = {}
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.predecessors(term):
                self.term_neighbor_map[term].append(child)
        i = 0
        while True:
            leaves = [n for n in dG.nodes() if dG.in_degree(n) == 0]
            # print(i, 'len', len(leaves))
            i += 1
            if len(leaves) == 0:
                break

            for term in leaves:
                input_size = 0
                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                term_hidden = self.term_dim_map[term]

                if input_size != 0:
                    self.add_module(term + '_linear_layer', nn.Linear(input_size, term_hidden))
                    self.add_module(term + '_act_layer0', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer1', nn.BatchNorm1d(term_hidden))
                    self.add_module(term + '_aux_linear_layer', nn.Linear(term_hidden, 2))
                    self.add_module(term + '_act_layer1', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer2', nn.BatchNorm1d(2))
                else:
                    self.add_module(term + '_linear_layer', nn.Linear(self.input_dim, term_hidden))
                    self.add_module(term + '_act_layer0', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer1', nn.BatchNorm1d(term_hidden))
                    self.add_module(term + '_aux_linear_layer', nn.Linear(term_hidden, 2))
                    self.add_module(term + '_act_layer1', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer2', nn.BatchNorm1d(2))

            self.term_layer_list.append(leaves)
            dG.remove_nodes_from(leaves)

        return self.term_layer_list

    def forward(self, x):

        term_NN_out_map = {}
        aux_out_map = {}
        for i, layer in enumerate(self.term_layer_list):
            for term in layer:
                if term in self.MIE_layer:
                    child_input = x
                else:
                    child_input_list = []
                    for child in self.term_neighbor_map[term]:
                        child_input_list.append(term_NN_out_map[child])
                    child_input = torch.cat(child_input_list, 1)

                term_NN_out = self._modules[term + '_act_layer0'](self._modules[term + '_linear_layer'](child_input))
                term_NN_out_map[term] = self._modules[term + '_batchnorm_layer1'](term_NN_out)
                aux_out = self._modules[term + '_act_layer1'](self._modules[term + '_aux_linear_layer'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term + '_batchnorm_layer2'](aux_out)

        return aux_out_map['final']



class AOP_NN_load(nn.Module):

    def __init__(self, layers_delete, dG, input_dim, num_hiddens_ratio):

        super(AOP_NN_load, self).__init__()
        self.layers_delete = layers_delete
        self.num_hiddens_ratio = num_hiddens_ratio
        self.cal_term_dim(dG)
        self.input_dim = input_dim
        self.MIE_layer = self.construct_MIE_layer(dG)
        self.term_layer_list = self.construct_NN_graph(dG)

    def cal_term_dim(self, dG):
        self.term_dim_map = {}
        for term in dG.nodes():
            term_degree = dG.degree(term)
            num_output = term_degree * self.num_hiddens_ratio
            num_output = int(num_output)
            self.term_dim_map[term] = num_output

    def construct_MIE_layer(self, dG):
        MIEs = [n for n in dG.nodes() if dG.in_degree(n) == 0]
        self.MIE_layer = MIEs
        # for term in self.MIE_layer:
        #     self.add_module(term + '_input_layer', nn.Linear(self.input_dim, self.term_dim_map[term]))
        return self.MIE_layer

    def construct_NN_graph(self, dG):

        self.term_layer_list = []
        self.term_neighbor_map = {}
        for term in dG.nodes():
            self.term_neighbor_map[term] = []
            for child in dG.predecessors(term):
                self.term_neighbor_map[term].append(child)

        dG.remove_nodes_from(self.layers_delete)
        i = 0
        while True:
            leaves = [n for n in dG.nodes() if dG.in_degree(n) == 0]
            # print(i, 'len', len(leaves))
            i += 1
            if len(leaves) == 0:
                break

            for term in leaves:
                input_size = 0
                for child in self.term_neighbor_map[term]:
                    input_size += self.term_dim_map[child]

                term_hidden = self.term_dim_map[term]

                if input_size != 0:
                    self.add_module(term + '_linear_layer', nn.Linear(input_size, term_hidden))
                    self.add_module(term + '_act_layer0', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer1', nn.BatchNorm1d(term_hidden))
                    self.add_module(term + '_aux_linear_layer', nn.Linear(term_hidden, 2))
                    self.add_module(term + '_act_layer1', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer2', nn.BatchNorm1d(2))
                else:
                    self.add_module(term + '_linear_layer', nn.Linear(self.input_dim, term_hidden))
                    self.add_module(term + '_act_layer0', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer1', nn.BatchNorm1d(term_hidden))
                    self.add_module(term + '_aux_linear_layer', nn.Linear(term_hidden, 2))
                    self.add_module(term + '_act_layer1', nn.Tanh())
                    self.add_module(term + '_batchnorm_layer2', nn.BatchNorm1d(2))

            self.term_layer_list.append(leaves)
            dG.remove_nodes_from(leaves)

        return self.term_layer_list

    def forward(self, term_NN_out_tensor):

        term_NN_out_map = {}

        m = 0
        n = 0
        for i, e in enumerate(self.layers_delete):
            n += self.term_dim_map[e]
            term_NN_out_map[e] = term_NN_out_tensor[:, m:n]
            m += self.term_dim_map[e]
        aux_out_map = {}
        for i, layer in enumerate(self.term_layer_list):
            for term in layer:

                child_input_list = []
                for child in self.term_neighbor_map[term]:
                    child_input_list.append(term_NN_out_map[child])
                child_input = torch.cat(child_input_list, 1)

                term_NN_out = self._modules[term+'_act_layer0'](self._modules[term+'_linear_layer'](child_input))
                term_NN_out_map[term] = self._modules[term + '_batchnorm_layer1'](term_NN_out)
                aux_out = self._modules[term+'_act_layer1'](self._modules[term + '_aux_linear_layer'](term_NN_out_map[term]))
                aux_out_map[term] = self._modules[term + '_batchnorm_layer2'](aux_out)

        return aux_out_map['final']


