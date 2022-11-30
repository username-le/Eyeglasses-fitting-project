#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import face_recognition
import matplotlib.image as mpimg
import os


# Dataframe of encodings of beautiful people in eyeglasses

# In[2]:


paths_men = []

folder = 'C:\\Users\\User\\TEACHME_Homeworks\\IMAGES\\men'

for root, dirs, files in os.walk(folder):
    for file in files:
        paths_men.append(os.path.join(root, file))
        


# In[7]:



data = pd.DataFrame(data = None, index = None, columns = ['file',  'encoding'])

for file in paths_men:
    image = face_recognition.load_image_file(file)
    face_encoding = face_recognition.face_encodings(image, model="large")
    face_enc_new = str(face_encoding)
    face_enc_new = face_enc_new.strip().strip("array([").strip("]").strip(")").strip("]")
    new_row = {'file':file,'encoding': face_enc_new}
    data = data.append(new_row, ignore_index=True)
    


# Data dataset contains filenames + raw encodings

# In[21]:


data.head(2)


# In[12]:


data['encoding'] = data['encoding'].str.replace('\n', '')
data['encoding'] = data['encoding'].str.replace(',', ' ')
data['encoding'] = data['encoding'].str.replace(']', '')
data['encoding'] = data['encoding'].str.replace(')', '')
data1 =  data['encoding'].str.split(expand=True)
df = data.join(data1)
df1 = df.iloc[:, 1:130].drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
foto_list = df1.index
df_ready = df.iloc[foto_list]
df_ready.to_csv('men_face_encodings_final.csv', sep=',', encoding='utf-8', index = False)


# In[2]:


dfff = pd.read_csv('men_face_encodings_final.csv', sep=',', encoding='utf-8')
numpy_rr = dfff.iloc[:, 2:130]
numpy_r = numpy_rr.astype(float)
n = numpy_r.to_numpy()


# Preparing the user's photo

# In[22]:


image_to_test = face_recognition.load_image_file('photo_5192842755284517152_y.jpg')
image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
plt.imshow(image_to_test);


# In[17]:


face_distances = face_recognition.face_distance(n, image_to_test_encoding)
dfff['face_dist'] = face_distances
sorted_df = dfff.sort_values(by='face_dist')
df_toshow_list_first10 = sorted_df[:10]['file'].to_list()
df_toshow_list_second10 = sorted_df[10:20]['file'].to_list()


# Results:

# In[19]:


face_dist_10 = sorted_df.face_dist[:10]
face_dist_10_20 = sorted_df.face_dist[10:20]

for file, number in zip(df_toshow_list_first10, face_dist_10):
    img = mpimg.imread(file)
    imgplot = plt.imshow(img)
    plt.show()
    print('% similarity is', round(100 - number *100, 2), '%')

    
for file, number in zip(df_toshow_list_second10, face_dist_10_20):
    img = mpimg.imread(file)
    imgplot = plt.imshow(img)
    plt.show()
    print('% similarity is', round(100 - number *100, 2), '%')

