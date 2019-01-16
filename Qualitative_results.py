#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


class unnormalize(torchvision.transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
ni = unnormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# In[17]:


data = torch.load('test_res_50.ckpt')
max_bleu_greedy = data['max_bleu_greedy']
min_bleu_greedy = data['min_bleu_greedy']
# max_bleu_beam = data['max_bleu_beam']
# min_bleu_beam = data['min_bleu_beam']

max_img = data['max_img']
min_img = data['min_img']

max_captions_pred_greedy = data['max_captions_pred_greedy']
min_captions_pred_greedy = data['min_captions_pred_greedy']
max_captions_target_greedy = data['max_captions_target_greedy']
min_captions_target_greedy =  data['min_captions_target_greedy']
# max_captions_pred_beam = data['max_captions_pred_beam']
# min_captions_pred_beam = data['min_captions_pred_beam']
# max_captions_target_beam = data['max_captions_target_beam']
# min_captions_target_beam =  data['min_captions_target_beam']


# In[18]:


npimg = ni(max_img[0][0]).numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))


# In[19]:


npimg = ni(max_img[1][0]).numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))


# In[20]:


npimg = ni(max_img[2][0]).numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))


# In[21]:


print(max_bleu_greedy)
print("TARGET")
print(' '.join(i for i in max_captions_target_greedy[0][0][0]))
print(' '.join(i for i in max_captions_target_greedy[1][0][0]))
print(' '.join(i for i in max_captions_target_greedy[2][0][0]))
print()
print("PREDITCED")
print(' '.join(i for i in max_captions_pred_greedy[0][0]))
print(' '.join(i for i in max_captions_pred_greedy[1][0]))
print(' '.join(i for i in max_captions_pred_greedy[2][0]))


# In[23]:


npimg = ni(min_img[0][0]).numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))


# In[24]:


npimg = ni(min_img[1][0]).numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))


# In[25]:


npimg = ni(min_img[2][0]).numpy()
plt.imshow(np.transpose(npimg, (1,2,0)))


# In[26]:


print(min_bleu_greedy)
print("TARGET")
print(' '.join(i for i in min_captions_target_greedy[0][0][0]))
print(' '.join(i for i in min_captions_target_greedy[1][0][0]))
print(' '.join(i for i in min_captions_target_greedy[2][0][0]))
print()
print("PREDITCED")
print(' '.join(i for i in min_captions_pred_greedy[0][0]))
print(' '.join(i for i in min_captions_pred_greedy[1][0]))
print(' '.join(i for i in min_captions_pred_greedy[2][0]))

