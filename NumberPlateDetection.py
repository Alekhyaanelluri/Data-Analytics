#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


img=cv2.imread('./img.jpg')
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(grey,cv2.COLOR_BGR2RGB))


# In[3]:


bfilter=cv2.bilateralFilter(grey,11,17,17)
edge=cv2.Canny(bfilter,30,200)
plt.imshow(cv2.cvtColor(edge,cv2.COLOR_BGR2RGB))


# In[4]:


import imutils


# In[5]:


keypnts=cv2.findContours(edge.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours=imutils.grab_contours(keypnts)
contours=sorted(contours,key=cv2.contourArea,reverse=True)[:10]


# In[6]:


location=None
for c in contours:
    aprox=cv2.approxPolyDP(c,10,True)
    if len(aprox)==4:
        location =aprox
        break


# In[7]:


print(location)


# In[8]:


mask=np.zeros(grey.shape,np.uint8)
newimage= cv2.drawContours(mask,[location],0,255,-1)
newimage= cv2.bitwise_and(img,img,mask=mask)
plt.imshow(cv2.cvtColor(newimage,cv2.COLOR_BGR2RGB))


# In[9]:


(x,y)=np.where(mask==255)
(x1,y1)=(np.min(x),np.min(y))
(x2,y2)=(np.max(x),np.max(y))
cropimg= grey[x1:x2+1,y1:y2+1]
plt.imshow(cv2.cvtColor(cropimg,cv2.COLOR_BGR2RGB))


# In[10]:


import easyocr


# In[11]:


reader=easyocr.Reader(['en'])
result= reader.readtext(cropimg)
result


# In[13]:


text=result[0][-2]+result[1][-2]+result[2][-2]+result[3][-2]
font=cv2.FONT_HERSHEY_SIMPLEX
print(text)
res=cv2.putText(img,text=text,org=(aprox[0][0][0],aprox[1][0][1]),fontFace=font,fontScale=2,color=(0,255,0),thickness=7,lineType=cv2.LINE_AA)
res=cv2.rectangle(img,tuple(aprox[0][0]),tuple(aprox[2][0]),(0,255,0))
plt.imshow(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))


# In[ ]:




