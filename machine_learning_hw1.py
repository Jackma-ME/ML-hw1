from PIL import Image
import tensorflow as tf
import numpy as np
import os
#----------------------------------------
def _store_image(img_path):
	num_list = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]
	label_list = []	
	img_list = []
	for num in range(10):
		for dpath, dname, fname in os.walk(img_path + "/Sample" + num_list[num]):
			for i in fname:
				img_list.append(os.path.join(dpath,i))
				lab_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
				lab_list[num] = 1
				label_list.append(lab_list)
	return img_list, label_list

def _int64_feature(data):
	return tf.train.Feature(int64_list = tf.train.Int64List(value=data))

def _bytes_feature(data):
	return tf.train.Feature(bytes_list = tf.train.BytesList(value=[data]))

def write_TFR(data, fea):
	(img_list, label_list) = _store_image(data)
	writer = tf.python_io.TFRecordWriter("TFRecords/" + data + ".tfrecords")
	for i in range(len(img_list)):
		img = Image.open(img_list[i])
		img = img.tobytes()
		label = label_list[i]
		feature = { fea + '/label':_int64_feature(label), fea + '/image':_bytes_feature(img)}
		features = tf.train.Features(feature = feature)
		example = tf.train.Example(features = features)
		writer.write(example.SerializeToString())
	writer.close()
	return 0
#---------------------------------------------------------
def read_TFR(data, fea):
	data_path = 'TFRecords/' + data + '.tfrecords'
	feature = {fea + '/image': tf.FixedLenFeature([], tf.string), fea + '/label': tf.FixedLenFeature([10], tf.int64)}
	filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features=feature)
	image = tf.decode_raw(features[ fea + '/image'], tf.uint8)
	label = tf.cast(features[ fea + '/label'], tf.int64)
	#image = tf.reshape(image, [128,128,1])
	image = tf.reshape(image, [128*128])
	#label = tf.reshape(label, [10])
	images, labels = tf.train.shuffle_batch([image, label], batch_size=2, capacity=30, num_threads=1, min_after_dequeue=10)
	return images, labels

#------------------------------------------------------
def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	# stride [1, x_movement, y_movement, 1]
	# Must have strides[0] = strides[3] = 1
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	# stride [1, x_movement, y_movement, 1]
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#----------------------------------------------------------
def BuildNetWork(xs, ys, keep_prob):
	x_image = tf.reshape(xs, [-1, 128, 128, 1])
	
	W_conv1 = weight_variable([5,5, 1,32], 'w_conv1') # patch 5x5, in size 1, out size 32
	b_conv1 = bias_variable([32], 'b_conv1')
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5,5, 32, 64], 'w_cov2') # patch 5x5, in size 32, out size 64
	b_conv2 = bias_variable([64], 'b_cov2')
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([32*32*64, 1024], 'w_fc1')
	b_fc1 = bias_variable([1024], 'b_fc1')

	h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([1024, 10], 'w_fc2')
	b_fc2 = bias_variable([10], 'b_fc2')
	
	prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))# loss
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	return prediction, train_step
#-------------------------------------------------------
def train(data_dir):
    # train your model with images from data_dir
    # the following code is just a placeholder
	train_img, train_label = read_TFR(data_dir, "train")
	xs = tf.placeholder(tf.float32, [None, 16384]) # 128x128
	ys = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)
	prediction, train_step = BuildNetWork(xs, ys, keep_prob)
	i = 0

	with tf.Session() as sess:
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			while not coord.should_stop():
				img, lab = sess.run([train_img, train_label])
				batch_xs, batch_ys = img, lab
				sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
				print(i,lab)
				i = i+1
		except tf.errors.OutOfRangeError:
			print("Done training")
		finally:
			saver = tf.train.Saver()
			saver.save(sess, "./ckpt/model.ckpt")
			coord.request_stop()
			coord.join(threads)
#-------------------------------------------------------
def test(data_dir):
    # make your model give prediction for images from data_dir
    # the following code is just a placeholder
	pre_list = []
	la_list = []
	val_img, val_label = read_TFR(data_dir, "val")
	xs = tf.placeholder(tf.float32, [None, 16384]) # 128x128
	ys = tf.placeholder(tf.float32, [None, 10])
	keep_prob = tf.placeholder(tf.float32)

	prediction, train_step = BuildNetWork(xs, ys, keep_prob)
	k = 0
	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, "./ckpt/model.ckpt")
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			while not coord.should_stop():
				img, lab = sess.run([val_img, val_label])
				batch_xs, batch_ys = img, lab
				pre = sess.run(prediction, feed_dict={xs: batch_xs, keep_prob: 1})
				k = k+1
				for i in range(2):
					for j in range(10):
						if pre[i][j] == 1:
							pre_list.append(j)
						if lab[i][j] == 1:
							la_list.append(j)
				print("i=",k, " lab= ", batch_ys, "pre= ", pre)
				
		except tf.errors.OutOfRangeError:
			print("Done validation")
		finally:
			coord.request_stop()
			coord.join(threads)
	print(len(pre_list), len(la_list))
	return pre_list, la_list
#---------------------------------------------------------------------------------
if __name__ == '__main__':
	print("test")
	write_TFR("training", "train")
	write_TFR("validation", "val")
	train("training")
	test("validation")

