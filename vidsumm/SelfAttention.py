import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    def __init__(self, Q_input_size, Q_output_size, F_input_size, F_output_size):
        super(SelfAttention, self).__init__()

        self.Q_m = Q_input_size
        self.Q_output_size = Q_output_size
                        
        self.F_m = F_input_size
        self.F_output_size = F_output_size

        self.Q_K = nn.Linear(in_features=self.Q_m, out_features=self.Q_output_size)
        self.Q_Q = nn.Linear(in_features=self.Q_m, out_features=self.Q_output_size)
        self.Q_V = nn.Linear(in_features=self.Q_m, out_features=self.Q_output_size)
        self.Q_output_linear = nn.Linear(in_features=self.Q_output_size, out_features=self.Q_m)

        self.F_K = nn.Linear(in_features=self.F_m, out_features=self.F_output_size)
        self.F_Q = nn.Linear(in_features=self.F_m, out_features=self.F_output_size)
        self.F_V = nn.Linear(in_features=self.F_m, out_features=self.F_output_size)
        self.F_output_linear = nn.Linear(in_features=self.F_output_size, out_features=self.F_m)
        
        self.drop = nn.Dropout(0.2)


    def forward(self, x, y):
        n1 = x.shape[0]
        n2 = y.shape[0]

        Q_K = self.Q_K(x)  
        Q_Q = self.Q_Q(x)  
        Q_V = self.Q_V(x)

        Q_att = torch.matmul(Q_Q.transpose(1,0), Q_K)

        Q_att_weights = nn.functional.softmax(Q_att, dim=-1)
        
        Q_weights = self.drop(Q_att_weights)
        
        Q_y = torch.matmul(Q_V, Q_weights.transpose(1,0))
        
        Q_y = self.Q_output_linear(Q_y)
        
        F_K = self.F_K(y)  
        F_Q = self.F_Q(y)  
        F_V = self.F_V(y)

        F_att = torch.matmul(F_Q.transpose(1,0), F_K)

        F_att_weights = nn.functional.softmax(F_att, dim=-1)
        
        F_weights = self.drop(F_att_weights)
        
        F_y = torch.matmul(F_V, F_weights.transpose(1,0))
        
        F_y = self.F_output_linear(F_y)


        return Q_y + x, Q_att_weights, F_y + y, F_att_weights