import tensorflow as tf
import numpy as np
import math 

#test

def model(input, img_size, img_channel):
# the structur of input : [batch, image data array]
	x = tf.reshape(input, [-1, img_size, img_size, img_channel])
	print(x)
	
	conv_size = 9
	conv_output_channel = 256
	stride = [1,2]
	for i in range(2):
		input_channel = x.get_shape().as_list()[-1]
		x = ConvNet(x, stride[i], conv_size, input_channel, conv_output_channel, "conv"+ str(i+1))
	x = CapsForm(x,8)
	x = CpasLayer(x, "PrimaryCaps", 32, 10, root_iter=3)
	x = CpasLayer(x, "DigitCaps", 10, 10, root_iter=0, is_output=True)
	return x
	
	
	

def ConvNet(x, stride, conv_size, input_channel, output_channel, name):
	std = math.sqrt(2/(conv_size*conv_size*output_channel))
	w_shape = [conv_size, conv_size, input_channel, output_channel]
	with tf.name_scope(name):
		weight = tf.Variable(tf.random_normal(w_shape,stddev=std), name="weight")
		conv = tf.nn.conv2d(x, weight, strides = [1, stride, stride, 1], padding = "VALID")
		return tf.nn.relu(conv)
			
def CapsForm(x, vec_length):
# x : convolution form
# vec_length: output capsule vector dimension
	x_length = x.get_shape().as_list()[-1]
	caps_nem = int(x.get_shape().as_list()[1]*x.get_shape().as_list()[1]*(x_length/vec_length))
	assert  x_length%vec_length ==0, "CapsForm(x,vec_length) input length not divisible by output length"
	
	x = tf.reshape(x, [-1, caps_nem, vec_length])
	return x
	

def Squashing(x):
# x : [batch_size, number of capsules, vector dimension]
	assert len(x.get_shape().as_list()) ==3, 'Squashing(x) input err'
	cap_size = x.get_shape().as_list()[1]
	norm = tf.norm(x, axis = -1)
	norm = norm/(1+norm*norm)
	norm = tf.reshape(norm, [-1, cap_size, 1])
	zero = tf.zeros(x.get_shape().as_list()[-1]) #
	return (norm+zero)*x

# In this implementation, PrimaryCaps has 32 weights.
# therefore, using tf.matmul is many operation and vary slow.
# This algrithm use element-wise multiply, reduce_sum and reshape
def Mul(x, caps_group_num, output_length):
# x : [batch_size, number of capsuls, vector dimension]
# caps_group_num ; number of capsule groups
	batch_size = x.get_shape().as_list()[0]
	x_length = x.get_shape().as_list()[-1]
	x_split = tf.split(x, num_or_size_splits=caps_group_num, axis = 1)
	x_split_caps_num = x_split[0].get_shape().as_list()[1]
	for i in range(caps_group_num):
		x_split[i] = tf.tile(x_split[i], (1,1,output_length))
	x_w_mul = tf.concat(x_split, 1)
	weight_shape = [x_length, output_length]
	w_std = math.sqrt(2/(output_length*x_length))
	with tf.variable_scope("weight"):
		w=[]
		for i in range(caps_group_num):
			#each cap group has a sharing weight
			w_i = tf.get_variable('w'+str(i), initializer = tf.random_normal(weight_shape,stddev=w_std), dtype = tf.float32) 
			w_i = tf.transpose(w_i)
			w_i = tf.split(w_i, num_or_size_splits=output_length, axis = 0)
			w_i = tf.concat(w_i,1)
			w_i = tf.tile(w_i,(x_split_caps_num,1))
			w.append(w_i)
		w = tf.concat(w,0)
	mul = x_w_mul*w
	mul = tf.split(mul, num_or_size_splits=output_length, axis = -1)
	for i in range(output_length):
		mul[i] = tf.reduce_sum(mul[i],axis=-1, keepdims=True)
	mul = tf.concat(mul, axis=-1)
	return mul

def Affine(x, b):
	new_caps_num = b.get_shape().as_list()[0]
	b_new = tf.split(b, new_caps_num, axis = 0)
	num = x.get_shape().as_list()[1]
	x_length = x.get_shape().as_list()[2]
	for i in range(len(b_new)):
		b_new[i] = tf.nn.softmax(b[i])
	b_new = tf.concat(b_new,0)
	b_new = tf.reshape(b_new, [num,1])
	x_result = x *b_new
	x_result = tf.split(x_result, new_caps_num, axis = 1)
	for i in range(new_caps_num):
		x_result[i] = tf.reduce_sum(x_result[i],axis = 1)
		x_result[i]= tf.reshape(x_result[i], [-1,1,x_length])
	return tf.concat(x_result,1)	
		
def RoutingIter(x_weight, b, name):
	output_caps = b.get_shape().as_list()[0]
	n = int(x_weight.get_shape().as_list()[1]/output_caps)
	with tf.variable_scope(name):
		x_new = Affine(x_weight, b)
		x_new = Squashing(x_new)	
		x_new = tf.split(x_new, output_caps, axis = 1)	
		for i in range(output_caps):
			x_new[i] = tf.tile(x_new[i], (1, n, 1))
		x_new = tf.concat(x_new, 1)
		dot = x_weight*x_new
		dot = tf.reduce_sum(dot,axis = -1)
		b_delta = tf.reduce_sum(dot,axis = 0)
		b_delta = tf.reshape(b_delta, [output_caps,n])
		
		return b+b_delta	

def CpasLayer(x, name, caps_group_num, output_caps, root_iter=0, is_output=False):	
	assert (is_output==False and root_iter>0) or (is_output==True)									
	assert len(x.get_shape().as_list()) == 3, 'Squashing(x) input dimension err'
	assert x.get_shape().as_list()[1]%caps_group_num == 0, 'Squashing(x) input parameter err'
	
	x_length = x.get_shape().as_list()[-1]
	caps_size = x.get_shape().as_list()[1]
	
	with tf.name_scope(name):
		with tf.name_scope("Squashing"):
			x = Squashing(x)
		
		if is_output==False:
			x_weight = []
			with tf.variable_scope("mul_weight"):
				for i in range(output_caps):
					with tf.variable_scope("capsule"+str(i+1)):
						x_weight.append(Mul(x, caps_group_num, 2*x_length))
				x_weight = tf.concat(x_weight,1)	
		
			with tf.variable_scope("Rooting"):
				b = tf.zeros([output_caps, caps_size], dtype=tf.float32)
				x_weight_stop = tf.stop_gradient(x_weight, name='stop_gradient') 
				for iter in range(root_iter):
					b = RoutingIter(x_weight_stop, b, "iter"+str(iter+1))
					
			x = Affine(x_weight, b)
			return x
		else:	
			return tf.norm(x, axis = -1)

#test
