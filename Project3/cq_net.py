'''cq_net.py
Implements the CQ network
CS443: Computational Neuroscience
Alice Cole Ethan
Project 3: Competitive Networks
'''
import numpy as np
import csv
import competitive_nets

class CQNet:
    '''
    decay = A
    capacity = B
    feedback_strength = D
    lower_bound = C

    '''
    def __init__(self, filepath="data/primacy_gradients.csv"):
        with open(filepath) as prim:
            rows = csv.reader(prim)
            for row in rows:
                x = row
        self.x = np.array(x, dtype="float32")
        self.y = np.copy(self.x)
        self.w = np.zeros(self.x.shape)
        # self.x = self.x.astype(np.float32)

    def working_mem(self, decay, capacity, feedback_strength):
        """This is  linear layer(xi)"""
        for i in range(self.x.shape[0]):
            left = - decay * self.x[i]
            left2 = (capacity - self.x[i]) * self.x[i] 
            right = self.x[i] * (competitive_nets.sum_not_I(self.x)[i] + feedback_strength * self.w[i])
            self.x[i] += left + left2 - right
        return np.array(self.x)
        

    def rcf_wta(self, decay, capacity, go_signal, lower_bound):
        """This is a winner take all layer (yi)"""
        for i in range(self.x.shape[0]):
            self.y[i] += -decay * self.y[i] + (capacity - self.y[i]) * (self.y[i]**2 + go_signal * self.x[i]) - (lower_bound + self.y[i]) * competitive_nets.sum_not_I(np.square(self.y))[i]
            # self.y[i] += -decay*I[i] + (capacity-I[i])*((I[i]**2)+ go_signal*I[i])-(lower_bound + I[i])*np.sum(np.square(not_i_I))
        
        return np.array(self.y)

    def inhibitory(self, decay, capacity, threshold):
        """ This is the inhibitory wi layer."""
        for i in range(self.x.shape[0]):
            self.w[i] += -decay * self.w[i] + (capacity - self.w[i]) * np.where(self.y[i]-threshold > 0, self.y[i]-threshold, 0)
        return self.w


    def competitive_queue(self, I, decay, capacity, feedback_strength, go_signal, lower_bound, threshold):
        """This puts together all the layers"""
        
        
        while np.sum(self.w) < 0.5:
            working_memory_output = self.working_mem(decay, capacity, feedback_strength)
            print("\nWorking Memory output:\n", working_memory_output)
            rcf_wta_output = self.rcf_wta(decay, capacity, go_signal, lower_bound)
            print("\nWinner Take all output:\n", rcf_wta_output)
            inhibitory_output = self.inhibitory(decay, capacity, threshold)
            print("\nInhibitory Output:\n", inhibitory_output)
        pass

    def get_x(self):
        return self.x


#main method
cq = CQNet()
cq.competitive_queue(I = cq.get_x(), decay = 1.0, capacity = 1.0, feedback_strength = 1.0, go_signal = 1, lower_bound = 0, threshold = .2)

