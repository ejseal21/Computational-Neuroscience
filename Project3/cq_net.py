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
    def __init__(self, num_gradients=6, filepath="data/primacy_gradients.csv", dt=0.01):
        with open(filepath) as prim:
            rows = csv.reader(prim)
            all_rows = []
            for row in rows:
                x = row
                all_rows.append(x)

        float_x = np.array(all_rows[num_gradients-1], dtype="float32")
        # self.threshold = (num_gradients*5)/100
        # print(self.threshold)
        self.x = float_x/np.sum(float_x)
        self.y = np.zeros(self.x.shape, dtype="float32")
        self.w = np.zeros(self.x.shape, dtype="float32")

        self.dt = dt


    def working_mem(self, decay, capacity, feedback_strength, threshold):
        """This is  linear layer(xi)"""

        left = - decay * self.x
        left2 = (capacity - self.x) * self.x 
        right = self.x * (competitive_nets.sum_not_I(self.x) + feedback_strength * self.w)
        self.x += (left + left2 - right) * self.dt
        return np.array(self.x)
        

    def rcf_wta(self, decay, capacity, go_signal, lower_bound):
        """This is a winner take all layer (yi)"""
        self.y += (-decay * self.y + (capacity - self.y) * (self.y**2 + go_signal * self.x) - (lower_bound + self.y) * competitive_nets.sum_not_I(np.square(self.y))) * self.dt
        return np.array(self.y)

    def inhibitory(self, decay, capacity, threshold):
        """ This is the inhibitory wi layer."""
        self.w += (-decay * self.w + (capacity - self.w) * np.where(self.y-threshold > 0, self.y-threshold, 0)) * self.dt

        return self.w


    def competitive_queue(self, I, decay_x, decay_y, decay_w, capacity_x, capacity_y, capacity_w, feedback_strength, go_signal, lower_bound, threshold):
        """This puts together all the layers"""
        x_hist = np.empty((1, self.x.size))
        y_hist = np.empty((1, self.y.size))
        w_hist = np.empty((1, self.w.size))

        while np.argmax(self.w)!=(self.w.shape[0]-1):
            #zero out anything that meets the threshold value
            self.w[np.where(self.y-threshold > 0)[0]] = 0
            self.x[np.where(self.y-threshold > 0)[0]] = 0
            self.x[np.where(self.y-threshold > 0)[0]] = 0
            working_memory_output = self.working_mem(decay_x, capacity_x, feedback_strength, threshold)
            x_hist = np.vstack((x_hist, np.copy(self.x)))
            # print("\nWorking Memory output:\n", working_memory_output)
            rcf_wta_output = self.rcf_wta(decay_y, capacity_y, go_signal, lower_bound)
            y_hist = np.vstack((y_hist, np.copy(self.y)))
            # print("\nWinner Take all output:\n", rcf_wta_output)
            inhibitory_output = self.inhibitory(decay_w, capacity_w, threshold)
            w_hist = np.vstack((w_hist, np.copy(self.w)))
            # print("\nInhibitory Output:\n", inhibitory_output)
            # print("\n------------------------------")
            if np.argwhere(self.w).size > 0:
                print(np.nonzero(self.w)[0])
        return x_hist, y_hist, w_hist

        

    def get_x(self):
        return self.x

    def get_w(self):
        return self.w
    
    def get_y(self):
        return self.y

#main method
cq = CQNet(num_gradients=10)
cq.competitive_queue(I = cq.get_x(), decay_x=0.5, decay_y=1, decay_w=.01, capacity_x=1.0, capacity_y=2.0, capacity_w=1.0, feedback_strength = 0, go_signal =1.9, lower_bound = 0, threshold = .55)


#EXTENSION 1
# cq = CQNet(num_gradients=14, filepath="data/primacy_gradients_ext.csv")
# cq.competitive_queue(I = cq.get_x(), decay_x=0.5, decay_y=1, decay_w=.01, capacity_x=1.0, capacity_y=2.0, capacity_w=1.0, feedback_strength = 0, go_signal =1.9, lower_bound = 0, threshold = .3)
