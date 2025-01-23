#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ProjectFunctions import *
import argparse


# In[ ]:


parser = argparse.ArgumentParser()

parser.add_argument("-f", type = str, default = 8)

parser.add_argument("-s", type = int, default = 8)

args = parser.parse_args()

filename = args.f
numseeds = args.s


# In[ ]:


#print(filename)


# In[ ]:


energydict = generate_sorted_classical_data(8, 14, range(numseeds))
save_dict_to_array_file(energydict, filename)

