#!/usr/bin/env python
# coding: utf-8

# ## Load Modules

# In[ ]:


import torch
import numpy as np
from sklearn.utils import shuffle
from Evaluation import evaluate
from Generation import generate
from Path import AF_FILE_PATH, NORMAL_FILE_PATH
# from NumpyToImage import numpy_to_image
# from FID import calculate_fid_score


# ## Used Device

# In[ ]:


print('cuda' if torch.cuda.is_available() else 'cpu')


# ## Load & Shuffle AF Samples

# In[ ]:


af_images = shuffle(np.load(AF_FILE_PATH))
print("af_images.shape: ", af_images.shape)


# ## Partition AF Samples

# In[ ]:


partitionIndex = af_images.shape[0] // 4
af_1 = af_images[:partitionIndex, :, :]
print("af1.shape: ", af_1.shape)
af_2 = af_images[partitionIndex : 2 * partitionIndex, :, :]
print("af2.shape: ", af_2.shape)
af_3 = af_images[2 * partitionIndex : 3 * partitionIndex + 1, :, :]
print("af3.shape: ", af_3.shape)
af_4 = af_images[3 * partitionIndex + 1 :, :, :]
print("af4.shape: ", af_4.shape)

af_123 = np.concatenate([af_1, af_2, af_3], axis=0)  # The 1st AF train-set
print("af_123.shape: ", af_123.shape)
af_124 = np.concatenate([af_1, af_2, af_4], axis=0)  # The 2nd AF train-set
print("af_124.shape: ", af_124.shape)
af_134 = np.concatenate([af_1, af_3, af_4], axis=0)  # The 3rd AF train-set
print("af_134.shape: ", af_134.shape)
af_234 = np.concatenate([af_2, af_3, af_4], axis=0)  # The 4th AF train-set
print("af_234.shape: ", af_234.shape)


# ## Augmentation Option

# In[ ]:


AUGMENTATION = True
print('Using Augmentation' if AUGMENTATION else 'No Augmentation')


# # Augment Training Folds One by One

# ### Fold 1

# In[ ]:


generated_123 = None
if AUGMENTATION:
    generated_123 = generate(af_123, 'eps_images/1')
    print("generated_123.shape: ", generated_123.shape)


# ### Fold 2

# In[ ]:


generated_124 = None
if AUGMENTATION:
    generated_124 = generate(af_124, 'eps_images/2')
    print("generated_124.shape: ", generated_124.shape)


# ### Fold 3

# In[ ]:


generated_134 = None
if AUGMENTATION:
    generated_134 = generate(af_134, 'eps_images/3')
    print("generated_134.shape: ", generated_134.shape)


# ### Fold 4

# In[ ]:


generated_234 = None
if AUGMENTATION:
    generated_234 = generate(af_234, 'eps_images/4')
    print("generated_234.shape: ", generated_234.shape)


# ## Load & Shuffle Normal Samples

# In[ ]:


normal_images = shuffle(
    np.load(NORMAL_FILE_PATH)
)  # Loads 360 x 360 preprocessed images of normal signals
print("normal_images.shape: ", normal_images.shape)


# ## Partition Normal Samples

# In[ ]:


partitionIndex = normal_images.shape[0] // 4
normal_1 = normal_images[:partitionIndex, :, :]
print("normal_1.shape: ", normal_1.shape)
normal_2 = normal_images[partitionIndex : 2 * partitionIndex, :, :]
print("normal_2.shape: ", normal_2.shape)
normal_3 = normal_images[2 * partitionIndex : 3 * partitionIndex, :, :]
print("normal_3.shape: ", normal_3.shape)
normal_4 = normal_images[3 * partitionIndex :, :, :]
print("normal_4.shape: ", normal_4.shape)

normal_123 = np.concatenate([normal_1, normal_2, normal_3], axis=0)
print("normal_123.shape: ", normal_123.shape)
normal_124 = np.concatenate([normal_1, normal_2, normal_4], axis=0)
print("normal_124.shape: ", normal_124.shape)
normal_134 = np.concatenate([normal_1, normal_3, normal_4], axis=0)
print("normal_134.shape: ", normal_134.shape)
normal_234 = np.concatenate([normal_2, normal_3, normal_4], axis=0)
print("normal_234.shape: ", normal_234.shape)


# ## Evaluate for All 4 Folds

# ### Fold 1

# In[ ]:


evaluate(
    train_normal=normal_123,
    train_af=af_123,
    val_normal=normal_4,
    val_af=af_4,
    generated_af=generated_123,
    showModelSummary=False,
    checkpoint_filepath="./checkpoints/my_checkpoint/best_4",
)


# ### Fold 2

# In[ ]:


evaluate(
    train_normal=normal_124,
    train_af=af_124,
    val_normal=normal_3,
    val_af=af_3,
    generated_af=generated_124,
    showModelSummary=False,
    checkpoint_filepath="./checkpoints/my_checkpoint/best_3",
)


# ### Fold 3

# In[ ]:


evaluate(
    train_normal=normal_134,
    train_af=af_134,
    val_normal=normal_2,
    val_af=af_2,
    generated_af=generated_134,
    showModelSummary=False,
    checkpoint_filepath="./checkpoints/my_checkpoint/best_2",
)


# ### Fold 4

# In[ ]:


evaluate(
    train_normal=normal_234,
    train_af=af_234,
    val_normal=normal_1,
    val_af=af_1,
    generated_af=generated_234,
    showModelSummary=False,
    checkpoint_filepath="./checkpoints/my_checkpoint/best_1",
)

