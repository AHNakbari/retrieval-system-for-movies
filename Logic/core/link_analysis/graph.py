import networkx as nx

class LinkGraph:
    """
    Use this class to implement the required graph in link analysis.
    You are free to modify this class according to your needs.
    You can add or remove methods from it.
    """
    def __init__(self):
        self.adj_list = {}
        self.rev_adj_list = {}

    def add_edge(self, u_of_edge, v_of_edge):
        if u_of_edge not in self.adj_list:
            self.adj_list[u_of_edge] = set()
        if v_of_edge not in self.rev_adj_list:
            self.rev_adj_list[v_of_edge] = set()
        self.adj_list[u_of_edge].add(v_of_edge)
        self.rev_adj_list[v_of_edge].add(u_of_edge)

    def add_node(self, node_to_add):
        if node_to_add not in self.adj_list:
            self.adj_list[node_to_add] = set()
        if node_to_add not in self.rev_adj_list:
            self.rev_adj_list[node_to_add] = set()

    def get_successors(self, node):
        return self.adj_list.get(node, set())

    def get_predecessors(self, node):
        return self.rev_adj_list.get(node, set())
