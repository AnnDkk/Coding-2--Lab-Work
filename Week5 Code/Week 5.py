#!/usr/bin/env python
# coding: utf-8

# In[29]:


list1=[12,25,8,2,16,4,10,6,10,20]
list2=[12,25,8,2,16,4,10,6,10,20]
count=1

while count<20:

   for i in range(len(list1)):
      if i==0:
         list2[i]=int(list1[i]/2+list1[len(list1)-1]/2)
         #print(list2)
      else:
         list2[i]=int(list1[i]/2+list1[i-1]/2)

   for i in range(len(list1)):
      if list2[i]%2==1:
         list2[i]=list2[i]+1

   for i in range(len(list1)):
      list1[i]=list2[i]
      
   print(list1)

   equal=0
   for i in range(len(list1)):
      if list1[0]==list2[i]:
         equal=equal+1
        
         
   if equal==len(list1):
      print('After',count,'passes you have',list1[0],'dumplings on your plate')
      break
   
   count=count+1


# In[4]:





# In[ ]:





# In[ ]:




