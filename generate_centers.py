import random
import numpy as np

def array(centers0,R,Lx,Ly,number_of_sides,number_of_length):
 R = Ly/(8.*number_of_sides) #radius
 s_y = Ly/number_of_sides #space between centers
 s_x = Lx/number_of_length #space between centers
 for i in range(number_of_length):
   for j in range(number_of_sides):
     centers0[i*number_of_sides+j] = [i*s_x+s_x/2.,j*s_y+s_y/2.]
 #print centers0[2]
 #temp = np.copy(centers0)
 #centers0[2] = temp[3]
 #centers0[3] = temp[2]
 #print  centers0[2]
 return
