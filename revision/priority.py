"""Finite nonpreemptive priority service for the classical-control baseline."""
from collections import deque
from itertools import count
from heapq import heappop, heappush

from network import BUFFER, Network


class PriorityNetwork(Network):
    def __init__(self, topo, outage=None):
        super().__init__(topo, outage)
        self.waiting = {edge: [] for edge in topo.c}
        self.busy = {edge: False for edge in topo.c}
        self.busy_until = {edge: 0. for edge in topo.c}
        self.arrival_order = count()

    def backlog(self, edge, now):
        end = max(now, self.busy_until[edge]) if self.busy[edge] else now
        end += sum(self.topo.c[edge].service(item[2].size) for item in self.waiting[edge])
        self.queues[edge] = deque([end]) if end > now else deque()

    def send(self, time, packet, hop=0):
        if hop == len(packet.route) - 1:
            if packet.kind == "background":
                self.bg_delivered += 1
            if packet.finish:
                packet.finish(time)
            return
        edge = packet.route[hop], packet.route[hop + 1]
        if self.busy[edge]:
            if len(self.waiting[edge]) + 1 >= BUFFER:
                self.drops[packet.kind] += 1
                return
            priority = int(packet.kind == "background")
            heappush(self.waiting[edge], (priority, next(self.arrival_order), packet, hop))
            self.backlog(edge, time)
        else:
            self.start(time, packet, hop, edge)

    def start(self, time, packet, hop, edge):
        self.busy[edge] = True
        self.bytes[packet.kind] += packet.size
        finish = time + self.topo.c[edge].service(packet.size)
        self.busy_until[edge] = finish
        self.at(finish, self.complete, packet, hop, edge)
        self.backlog(edge, time)

    def complete(self, time, packet, hop, edge):
        lost = False
        if self.outage and packet.kind != "background":
            group, begin, end = self.outage
            lost = self.topo.c[edge].risk == group and begin <= time < end
        if lost:
            self.drops[packet.kind] += 1
        else:
            self.at(time + self.topo.c[edge].propagation, self.send, packet, hop + 1)
        self.busy[edge] = False
        if self.waiting[edge]:
            _, _, following, next_hop = heappop(self.waiting[edge])
            self.start(time, following, next_hop, edge)
        else:
            self.backlog(edge, time)
