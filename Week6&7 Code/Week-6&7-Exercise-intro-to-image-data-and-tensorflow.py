#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Load the os library
import os

# Load the request module
import urllib.request

# Import SSL which we need to setup for talking to the HTTPS server
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Create a directory
try: 
    os.mkdir('img_align_celeba')

    # Now perform the following 100 times:
    for img_i in range(1, 101):

        # create a string using the current loop counter
        f = '000%03d.jpg' % img_i

        # and get the url with that string appended the end
        url = 'https://s3.amazonaws.com/cadl/celeb-align/' + f

        # We'll print this out to the console so we can see how far we've gone
        print(url, end='\r')

        # And now download the url to a location inside our new directory
        urllib.request.urlretrieve(url, os.path.join('img_align_celeba', f))
except:
    #os.rm('img_align_celeba')
    print("You may need to delete the existing 'img_align_celeba' folder in your directory")


# #Intro to image datasets and tensorflow
# - This tutorial is a version of Parag Mital's CADL session 1
# - I've fixed it up so that it will run on your systems
# - It demonstrates how to make a large dataset of images
# - And shows you how to process them easily in NUMPY
# - It also shows you how to do simple calculations in tensorflow
# - This will really help you understand what is going on when you use TF
# - So don't skip it.

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# I'll be using a popular image dataset for faces called the CelebFaces dataset.  I've provided some helper functions which you can find on the resources page, which will just help us with manipulating images and loading this dataset. 
# 
# # NOTE - If you want to run this more than once, you may want to delete the downloaded images - they will be in the same folder that this notebook is in.

# Let's get the 50th image in this list of files, and then read the file at that location as an image, setting the result to a variable, `img`, and inspect a bit further what's going on:

# In[12]:


files = os.listdir('img_align_celeba')# img.<tab>
import matplotlib.pyplot as plt
import numpy as np

print(os.path.join('img_align_celeba', files[0]))
plt.imread(os.path.join('img_align_celeba', files[0]))

files = [os.path.join('img_align_celeba', file_i)
 for file_i in os.listdir('img_align_celeba')
 if '.jpg' in file_i]

# There should be 100 files, with the last one being number 99

img = plt.imread(files[99])

print(img)


# When I print out this image, I can see all the numbers that represent this image.  We can use the function `imshow` to see this:

# In[13]:


# If nothing is drawn and you are using notebook, try uncommenting the next line:
#%matplotlib inline
plt.imshow(img)


# <a name="understanding-image-shapes"></a>
# ## Understanding Image Shapes
# 
# Let's break this data down a bit more.  We can see the dimensions of the data using the `shape` accessor:

# In[14]:


img.shape
# (218, 178, 3)


# This means that the image has 218 rows, 178 columns, and 3 color channels corresponding to the Red, Green, and Blue channels of the image, or RGB.  Let's try looking at just one of the color channels.

# In[15]:


plt.imshow(img[:, :, 0], cmap='gray') # Red Channel


# In[16]:


plt.imshow(img[:, :, 1], cmap='gray') # Green Channel


# In[17]:


plt.imshow(img[:, :, 2], cmap='gray') # Blue Channel


# We use the special colon operator to say take every value in this dimension.  This is saying, give me every row, every column, and the 0th dimension of the color channels.  What we're seeing is the amount of Red, Green, or Blue contributing to the overall color image.
# 
# Let's use another helper function which will load every image file in the celeb dataset rather than just give us the filenames like before.  By default, this will just return the first 100 images because loading the entire dataset is a bit cumbersome.  In one of the later sessions, I'll show you how tensorflow can handle loading images using a pipeline so we can load this same dataset.  For now, let's stick with this:

# In[18]:


imgs = [plt.imread(files[file_i])
        for file_i in range(100)]

#imgs = utils.get_celeb_imgs() # nope nope nope


# We now have a list containing our images.  Each index of the `imgs` list is another image which we can access using the square brackets:

# In[19]:


plt.imshow(imgs[88])


# <a name="the-batch-dimension"></a>
# ## The Batch Dimension
# 
# Remember that an image has a shape describing the height, width, channels:

# In[20]:


imgs[0].shape


# It turns out we'll often use another convention for storing many images in an array using a new dimension called the batch dimension.  The resulting image shape will be exactly the same, except we'll stick on a new dimension on the beginning... giving us number of images x the height x the width x the number of color channels.
# 
# N x H x W x C
# 
# A Color image should have 3 color channels, RGB.
# 
# We can combine all of our images to have these 4 dimensions by telling numpy to give us an array of all the images.

# In[21]:


data = np.array(imgs) # make 'data' = our numpy array
data.shape
print(data.shape)
print("The shape of our new 'data' object is a 'batch' of 100 images, with a height of 218, width of 178, and 3 colour channels")
print("If your images aren't all the same size to begin with, then this won't work!")


# This will only work if every image in our list is exactly the same size.  So if you have a wide image, short image, long image, forget about it.  You'll need them all to be the same size.  If you are unsure of how to get all of your images into the same size, then please please refer to the online resources for the notebook I've provided which shows you exactly how to take a bunch of images of different sizes, and crop and resize them the best we can to make them all the same size.
# 
# <a name="meandeviation-of-images"></a>
# ## Mean/Deviation of Images
# 
# Now that we have our data in a single numpy variable, we can do alot of cool stuff.  Let's look at the mean of the batch channel:

# In[25]:


mean_img = np.mean(data, axis=0) # This is the mean of the 'batch' channel
plt.imshow(mean_img.astype(np.uint8))
print("look at this average person")


# This is the first step towards building our robot overlords.  We've reduced down our entire dataset to a single representation which describes what most of our dataset looks like.  There is one other very useful statistic which we can look at very easily:

# In[29]:


std_img = np.std(data, axis=0)
plt.imshow(std_img.astype(np.uint8))
print("This is the standard deviation - the variance of the mean")


# So this is incredibly cool.  We've just shown where changes are likely to be in our dataset of images.  Or put another way, we're showing where and how much variance there is in our previous mean image representation.
# 
# We're looking at this per color channel.  So we'll see variance for each color channel represented separately, and then combined as a color image.  We can try to look at the average variance over all color channels by taking their mean:

# In[33]:


plt.imshow(np.mean(std_img, axis=2).astype(np.uint8)) # Mean of all colour channels
print("Mean of all colour channels")


# This is showing us on average, how every color channel will vary as a heatmap.  The more red, the more likely that our mean image is not the best representation.  The more blue, the less likely that our mean image is far off from any other possible image.
# 
# ## Dataset Preprocessing
# 
# Think back to when I described what we're trying to accomplish when we build a model for machine learning?  We're trying to build a model that understands invariances.  We need our model to be able to express *all* of the things that can possibly change in our data.  Well, this is the first step in understanding what can change.  If we are looking to use deep learning to learn something complex about our data, it will often start by modeling both the mean and standard deviation of our dataset.  We can help speed things up by "preprocessing" our dataset by removing the mean and standard deviation.  What does this mean?  Subtracting the mean, and dividing by the standard deviation.  Another word for that is "normalization".
# 
# ## Histograms
# 
# Let's have a look at our dataset another way to see why this might be a useful thing to do.  We're first going to convert our `batch` x `height` x `width` x `channels` array into a 1 dimensional array.  Instead of having 4 dimensions, we'll now just have 1 dimension of every pixel value stretched out in a long vector, or 1 dimensional array.

# In[34]:


flattened = data.ravel()
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
print(data[:1])
print(flattened[:10])


# We first convert our N x H x W x C dimensional array into a 1 dimensional array.  The values of this array will be based on the last dimensions order.  So we'll have: [<font color='red'>251</font>, <font color='green'>238</font>, <font color='blue'>205</font>, <font color='red'>251</font>, <font color='green'>238</font>, <font color='blue'>206</font>, <font color='red'>253</font>, <font color='green'>240</font>, <font color='blue'>207</font>, ...]
# 
# We can visualize what the "distribution", or range and frequency of possible values are.  This is a very useful thing to know.  It tells us whether our data is predictable or not.

# In[35]:


plt.hist(flattened.ravel(), 255)


# The last line is saying give me a histogram of every value in the vector, and use 255 bins.  Each bin is grouping a range of values.  The bars of each bin describe the frequency, or how many times anything within that range of values appears.In other words, it is telling us if there is something that seems to happen more than anything else.  If there is, it is likely that a neural network will take advantage of that.
# 
# 
# <a name="histogram-equalization"></a>
# ## Histogram Equalization
# 
# The mean of our dataset looks like this:

# In[36]:


plt.hist(mean_img.ravel(), 255)


# When we subtract an image by our mean image, we remove all of this information from it.  And that means that the rest of the information is really what is important for describing what is unique about it.
# 
# Let's try and compare the histogram before and after "normalizing our data":

# In[48]:


bins = 25
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
axs[0].hist((data[0]).ravel(), bins)
axs[0].set_title('img distribution')
axs[1].hist((mean_img).ravel(), bins)
axs[1].set_title('mean distribution')
axs[2].hist((data[0] - mean_img).ravel(), bins)
axs[2].set_title('(img - mean) distribution')


# What we can see from the histograms is the original image's distribution of values from 0 - 255.  The mean image's data distribution is mostly centered around the value 100.  When we look at the difference of the original image and the mean image as a histogram, we can see that the distribution is now centered around 0.  What we are seeing is the distribution of values that were above the mean image's intensity, and which were below it.  Let's take it one step further and complete the normalization by dividing by the standard deviation of our dataset:

# In[47]:


fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
axs[0].hist((data[0] - mean_img).ravel(), bins)
axs[0].set_title('(img - mean) distribution')
axs[1].hist((std_img).ravel(), bins)
axs[1].set_title('std deviation distribution')
axs[2].hist(((data[0] - mean_img) / std_img).ravel(), bins)
axs[2].set_title('((img - mean) / std_dev) distribution')


# Now our data has been squished into a peak!  We'll have to look at it on a different scale to see what's going on:

# In[49]:


axs[2].set_xlim([-150, 150])
axs[2].set_xlim([-100, 100])
axs[2].set_xlim([-50, 50])
axs[2].set_xlim([-10, 10])
axs[2].set_xlim([-5, 5])


# What we can see is that the data is in the range of -3 to 3, with the bulk of the data centered around -1 to 1.  This is the effect of normalizing our data: most of the data will be around 0, where some deviations of it will follow between -3 to 3.
# 
# If our data does not end up looking like this, then we should either (1): get much more data to calculate our mean/std deviation, or (2): either try another method of normalization, such as scaling the values between 0 to 1, or -1 to 1, or possibly not bother with normalization at all.  There are other options that one could explore, including different types of normalization such as local contrast normalization for images or PCA based normalization but we won't have time to get into those in this course.
# 
# <a name="tensorflow-basics"></a>
# # Tensorflow Basics
# 
# Let's now switch gears and start working with Google's Library for Numerical Computation, TensorFlow.  This library can do most of the things we've done so far.  However, it has a very different approach for doing so.  And it can do a whole lot more cool stuff which we'll eventually get into.  The major difference to take away from the remainder of this session is that instead of computing things immediately, we first define things that we want to compute later using what's called a `Graph`.  Everything in Tensorflow takes place in a computational graph and running and evaluating anything in the graph requires a `Session`.  Let's take a look at how these both work and then we'll get into the benefits of why this is useful:
# 
# <a name="variables"></a>
# ## Variables
# 
# We're first going to import the tensorflow library:

# In[53]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Let's take a look at how we might create a range of numbers.  Using numpy, we could for instance use the linear space function:

# In[54]:


x = np.linspace(-3.0, 3.0, 100)

# Immediately, the result is given to us.  An array of 100 numbers equally spaced from -3.0 to 3.0.
print(x)

# We know from numpy arrays that they have a `shape`, in this case a 1-dimensional array of 100 values
print(x.shape)

# and a `dtype`, in this case float64, or 64 bit floating point values.
print(x.dtype)


# <a name="tensors"></a>
# ## Tensors
# 
# In tensorflow, we could try to do the same thing using their linear space function:

# In[55]:


x = tf.linspace(-3.0, 3.0, 100)
print(x)


# Instead of a `numpy.array`, we are returned a `tf.Tensor`.  The name of it is "LinSpace/Slice:0".  Wherever we see this colon 0, that just means the output of.  So the name of this Tensor is saying, the output of LinSpace.
# 
# Think of `tf.Tensor`s the same way as you would the `numpy.array`.  It is described by its `shape`, in this case, only 1 dimension of 100 values.  And it has a `dtype`, in this case, `float32`.  But *unlike* the `numpy.array`, there are no values printed here!  That's because it actually hasn't computed its values yet.  Instead, it just refers to the output of a `tf.Operation` which has been already been added to Tensorflow's default computational graph.  The result of that operation is the tensor that we are returned.
# 
# <a name="graphs"></a>
# ## Graphs
# 
# Let's try and inspect the underlying graph.  We can request the "default" graph where all of our operations have been added:

# In[56]:


g = tf.get_default_graph()


# <a name="operations"></a>
# ## Operations
# 
# And from this graph, we can get a list of all the operations that have been added, and print out their names:

# In[57]:


[op.name for op in g.get_operations()]


# So Tensorflow has named each of our operations to generally reflect what they are doing.  There are a few parameters that are all prefixed by LinSpace, and then the last one which is the operation which takes all of the parameters and creates an output for the linspace.
# 
# <a name="tensor"></a>
# ## Tensor
# 
# We can request the output of any operation, which is a tensor, by asking the graph for the tensor's name:

# In[58]:


g.get_tensor_by_name('linspace/Slice' + ':0')


# What I've done is asked for the `tf.Tensor` that comes from the operation "LinSpace".  So remember, the result of a `tf.Operation` is a `tf.Tensor`.  Remember that was the same name as the tensor `x` we created before.
# 
# <a name="sessions"></a>
# ## Sessions
# 
# In order to actually compute anything in tensorflow, we need to create a `tf.Session`.  The session is responsible for evaluating the `tf.Graph`.  Let's see how this works:

# In[59]:


# We're first going to create a session:
sess = tf.Session()

# Now we tell our session to compute anything we've created in the tensorflow graph.
computed_x = sess.run(x)
print(computed_x)

# Alternatively, we could tell the previous Tensor to evaluate itself using this session:
computed_x = x.eval(session=sess)
print(computed_x)

# We can close the session after we're done like so:
sess.close()


# We could also explicitly tell the session which graph we want to manage:

# In[60]:


sess = tf.Session(graph=g)
sess.close()


# By default, it grabs the default graph.  But we could have created a new graph like so:

# In[61]:


g2 = tf.Graph()


# And then used this graph only in our session.
# 
# To simplify things, since we'll be working in iPython's interactive console, we can create an `tf.InteractiveSession`:

# In[62]:


sess = tf.InteractiveSession()
x.eval()


# Now we didn't have to explicitly tell the `eval` function about our session.  We'll leave this session open for the rest of the lecture.
# 
# <a name="tensor-shapes"></a>
# ## Tensor Shapes

# In[63]:


# We can find out the shape of a tensor like so:
print(x.get_shape())

# %% Or in a more friendly format
print(x.get_shape().as_list())


# <a name="many-operations"></a>
# ## Many Operations
# 
# Lets try a set of operations now.  We'll try to create a Gaussian curve.  This should resemble a normalized histogram where most of the data is centered around the mean of 0.  It's also sometimes refered to by the bell curve or normal curve.

# In[74]:


# The 1 dimensional gaussian takes two parameters, the mean value, and the standard deviation, which is commonly denoted by the name sigma.
mean = 0.0
sigma = 1.0

# Don't worry about trying to learn or remember this formula.  I always have to refer to textbooks or check online for the exact formula.
z = (tf.exp(tf.negative(tf.pow(x - mean,1.0) /
                   (3.0 * tf.pow(sigma, 3.0)))) *
     (2.0 / (sigma * tf.sqrt(3.0 * 3.1415))))


# Just like before, amazingly, we haven't actually computed anything.  We *have just added a bunch of operations to Tensorflow's graph.  Whenever we want the value or output of this operation, we'll have to explicitly ask for the part of the graph we're interested in before we can see its result.  Since we've created an interactive session, we should just be able to say the name of the Tensor that we're interested in, and call the `eval` function:

# In[75]:


res = z.eval()
plt.plot(res)
# # if nothing is drawn, and you are using ipython notebook, uncomment the next two lines:
# %matplotlib inline
# plt.plot(res)


# # Convolution
# 
# ## Creating a 2-D Gaussian Kernel
# 
# Let's try creating a 2-dimensional Gaussian - basically a 2D version of what we just did.  This can be done by multiplying a vector by its transpose.  If you aren't familiar with matrix math, I'll review a few important concepts.  This is about 98% of what neural networks do so if you're unfamiliar with this, then please stick with me through this and it'll be smooth sailing.  First, to multiply two matrices, their inner dimensions must agree, and the resulting matrix will have the shape of the outer dimensions.
# 
# So let's say we have two matrices, X and Y.  In order for us to multiply them, X's columns must match Y's rows.  I try to remember it like so:
# <pre>
#     (X_rows, X_cols) x (Y_rows, Y_cols)
#       |       |           |      |
#       |       |___________|      |
#       |             ^            |
#       |     inner dimensions     |
#       |        must match        |
#       |                          |
#       |__________________________|
#                     ^
#            resulting dimensions
#          of matrix multiplication
# </pre>
# But our matrix is actually a vector, or a 1 dimensional matrix.  That means its dimensions are N x 1.  So to multiply them, we'd have:
# <pre>
#      (N,      1)    x    (1,     N)
#       |       |           |      |
#       |       |___________|      |
#       |             ^            |
#       |     inner dimensions     |
#       |        must match        |
#       |                          |
#       |__________________________|
#                     ^
#            resulting dimensions
#          of matrix multiplication
# </pre>

# In[76]:


ksize = z.get_shape().as_list()[0]

z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))

plt.imshow(z_2d.eval())


# ## Your Tasks
# 
# ### Make a version of the Notebook with at least one major difference that you have introduced yourself.
# 
# * First, you must do some transformation on the image dataset that isn't included in the above document. You must use numpy to do this transformation.
# 
# * If you manage to do this, your next task is to collect and process your own dataset instead of the one provided. 
