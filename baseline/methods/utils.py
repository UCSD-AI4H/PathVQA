import numpy as np
from scipy import misc
import tensorflow as tf


def load_image_array(image_file):
	img = misc.imread(image_file)
	# GRAYSCALE
	if len(img.shape) == 2:
		img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
		img_new[:,:,0] = img
		img_new[:,:,1] = img
		img_new[:,:,2] = img
		img = img_new
	img_resized = misc.imresize(img, (224, 224))
	return (img_resized/255.0).astype('float32')


def extract_fc7_features(image_path, model_path):
	vgg_file = open(model_path,'rb')
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)
	images = tf.placeholder("float32", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })
	graph = tf.get_default_graph()

	sess = tf.Session()
	image_array = load_image_array(image_path)
	image_feed = np.ndarray((1,224,224,3))
	image_feed[0:,:,:] = image_array[:,:,:3]
	feed_dict  = { images : image_feed }
	fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
	fc7_features = sess.run(fc7_tensor, feed_dict = feed_dict)
	sess.close()
	return fc7_features


def extract_cnn7_features(image_path, model_path):
	vgg_file = open(model_path,'rb')
	vgg16raw = vgg_file.read()
	vgg_file.close()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(vgg16raw)
	images = tf.placeholder("float32", [None, 224, 224, 3])
	tf.import_graph_def(graph_def, input_map={ "images": images })
	graph = tf.get_default_graph()

	sess = tf.Session()
	image_array = load_image_array(image_path)
	image_feed = np.ndarray((1,224,224,3))
	image_feed[0:,:,:] = image_array[:,:,:3]
	feed_dict  = { images : image_feed }
	cnn7_tensor = graph.get_tensor_by_name("import/pool5:0")
	cnn7_features = sess.run(cnn7_tensor, feed_dict = feed_dict)
	sess.close()
	cnn7_features = np.transpose(cnn7_features,[0,3,1,2])
	cnn7_features = cnn7_features.reshape((1,512,49))
	return cnn7_features