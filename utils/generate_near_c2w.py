import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GetNearC2W:
    def __init__(self, args):
        super(GetNearC2W, self).__init__()
        self.near_c2w_type = args.near_c2w_type
        self.near_c2w_rot = args.near_c2w_rot
        self.near_c2w_trans = args.near_c2w_trans
        
        self.smoothing_rate = args.smoothing_rate
        self.smoothing_step_size = args.smoothing_step_size

    
    def __call__(self, c2w, all_poses=None,j=None,iter_=None):
        assert (c2w.shape == (3,4))
        
        if self.near_c2w_type == 'rot_from_origin':
            return self.rot_from_origin(c2w,iter_)
        elif self.near_c2w_type == 'near':
            return self.near(c2w, all_poses)
        elif self.near_c2w_type == 'random_pos':
            return self.random_pos(c2w)
        elif self.near_c2w_type == 'random_dir':
            return self.random_dir(c2w, j)
   
    def random_pos(self, c2w):     
        c2w[:, -1:] += self.near_c2w_trans*torch.randn(3).unsqueeze(-1)
        return c2w 
    
    def random_dir(self, c2w, j):
        rot = c2w[:3,:3]
        pos = c2w[:3,-1:]
        rot_theta, rot_phi = self.get_rotation_matrix(j)
        rot = torch.mm(rot_phi, rot)
        c2w = torch.cat((rot, pos), -1)
        return c2w
    
    def rot_from_origin(self, c2w,iter_=None):
        rot = c2w[:3,:3]
        pos = c2w[:3,-1:]
        rot_mat = self.get_rotation_matrix(iter_)
        pos = torch.mm(rot_mat, pos)
        rot = torch.mm(rot_mat, rot)
        c2w = torch.cat((rot, pos), -1)
        return c2w

    def get_rotation_matrix(self,iter_=None):
        #if iter_ is not None:
        #    rotation = self.near_c2w_rot * (self.smoothing_rate **(int(iter_/self.smoothing_step_size)))
        #else: 
        rotation = self.near_c2w_rot

        phi = (rotation*(np.pi / 180.))
        x = np.random.uniform(-phi, phi)
        y = np.random.uniform(-phi, phi)
        z = np.random.uniform(-phi, phi)
        
        rot_x = torch.Tensor([
                    [1,0,0],
                    [0,np.cos(x),-np.sin(x)],
                    [0,np.sin(x), np.cos(x)]
                    ])
        rot_y = torch.Tensor([
                    [np.cos(y),0,-np.sin(y)],
                    [0,1,0],
                    [np.sin(y),0, np.cos(y)]
                    ])
        rot_z = torch.Tensor([
                    [np.cos(z),-np.sin(z),0],
                    [np.sin(z),np.cos(z),0],
                    [0,0,1],
                    ])
        rot_mat = torch.mm(rot_x, torch.mm(rot_y, rot_z))
        return rot_mat
    

def get_near_pixel(coords, padding):
    '''
    padding is the distance range (manhattan distance)

    '''
    N = coords.size(0)
    m_distance = np.random.randint(1, padding+1) #manhattan distance
    # get diff 
    x_distance = torch.randint(0, m_distance+1, (N,1))
    y_distance = m_distance - x_distance
    sign_ = torch.randint(0,2,(N,2))*2 -1
    delta = torch.cat((x_distance, y_distance), dim=1)
    delta *= sign_
    # get near coords
    near_coords = coords + delta
    return near_coords
