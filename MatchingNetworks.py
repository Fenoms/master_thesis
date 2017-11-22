import tensorflow as tf 
import tensorflow.contrib.rnn as rnn
from tensorflow.python.ops.nn_ops import max_pool

class BidirectionalLSTM:
	def __init__(self, layer_sizes, batch_size):
		"""
		initialize a multi-layers bidirectional LSTM
		:param layer_sizes: A list containing the neuron numberd per layer
		:param batch_size: the experiment batch size
		"""

		self.reuse = False
		self.batch_size = batch_size
		self.layer_sizes = layer_sizes

	def __call__(self, inputs, name, training_flag=False):
		"""
		Runs the bidirectional LSTM, produces outputs and saves both forward and
		backward states as well as gradients
		:param inputs: the inputs should be a list of shape [sequence_length, batch_size, 64]
		:param name: the name of the operation
		:param training_flag:
		:return: returns the LSTM outputs, as well as the forward and backward hidden states
		"""

		with tf.name_scope('bid-lstm' + name), tf.variable_scope('bid-lstm', reuse = self.reuse):
			with tf.variable_scope("encoder"):
				fw_lstm_cells_encoder = [rnn.LSTMCell(num_units = self.layer_sizes[i], activation = tf.nn.tanh)
											for i in range(len(self.layer_sizes))]

				bw_lstm_cells_encoder = [rnn.LSTMCell(num_units = self.layer_sizes[i], activation = tf.nn.tanh)
											for i in range(len(self.layer_sizes))]


				outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
					fw_lstm_cells_encoder,
					bw_lstm_cells_encoder,
					inputs,
					dtype = tf.float32
					)

			print("output_shape:", tf.stack(outputs, axis=0).get_shape().as_list())su

			with tf.variable_scope("decoder"):
				fw_lstm_cells_decoder = [rnn.LSTMCell(num_units = self.layer_sizes[i], activation = tf.nn.tanh), 
											for i in range(len(self.layer_sizes))]
				bw_lstm_cells_decoder = [rnn.LSTMCell(num_units = self.layer_sizes[i], activation = tf.nn.tanh),
											for i in range(len(layer_sizes))]

				outputs, output_state_fw, output_state_bw = rnn.stack_bidirectional_rnn(
					fw_lstm_cells_decoder, 
					bw_lstm_cells_decoder, 
					outputs, 
					dtype = tf.float32
					)


		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKesys.TRAINABLE_VARIABLES, scope = 'bid-lstm')
		return outputs, output_state_fw, output_state_bw



class DistanceNetwork:
	def __init__(self):
		self.reuse = False



	def __call__(self, support_set, target_image, name, training_flag = False):
		"""
		this module calculates the cosine distance between each of the support set embeddings and the target image embddings
		:param support_set:  the embedding of support set images, tensor of shape [sequence_length, batch_size, 64]
		:param target image: the embedding target image, tensor of shape [batch_size, 64]
		:name: the name of this op appear on the graph
		:param training_flag: flag indicates training or evaluation (True/False)
		:return : A tensor cosine similarities of shape [batch_size, sequence_lewitngth, 1]
		"""

		with tf.name_scopee('distance_module' + name), tf.variable_scope('distance_module', reuse = self.reuse):
			eps = 1e-10
			similarities = []
			for support_image in tf.unstack(support_set, axis=0):
				sum_support = tf.reduce_sum(tf.square(support_image), 1, keep_dims = True)
				support_magnitude = tf.rsqrt(tf.clip_by_value(sum_support, eps, float("inf")))




class EmbeddingNetwork:
	def __init__(self, batch_size, layer_sizes):
		"""
		build a cnn to produce embeddings
		:param batch_size: batch_size for the experiment
		:param layer_sizes: a list of filter number in each layer
		"""

		self.reuse = False
		self.batch_size = batch_size
		self.layer_sizes = layer_sizes

	def _res_block(self, ):

	def __call__(self, image_input, training_flag = False, keep_prob = 1.0):
		"""
		run the cnn to produce the embeddings and the graidents
		:param image_input: the image to be embedded
		:param training_flag: indicate if in training step
		:param keep_prob: the dropout percentage in dropout layer
		:return : Embeddings of size [batch_size, 512]
		"""

		def leaky_relu(x, leak = 0.1, name = ''):
			return tf.maximum(x, x*leak, name = name)


		with tf.variable_scope('g', reuse = self.reuse):
			with tf.variable_scope('res_block_1'):
				short_cut = image_input
				nb_channels = self.layer_sizes[0]
				with tf.variable_scope('conv_block_1'):
					g_conv1_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv1_embedding = tf.contrib.layers.batch_norm(g_conv1_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv1_embedding = leaky_relu(g_conv1_embedding, name = 'outputs')
				with tf.variable_scope('conv_block_2'):
					g_conv1_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv1_embedding = tf.contrib.layers.batch_norm(g_conv1_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv1_embedding = leaky_relu(g_conv1_embedding, name = 'outputs')

				with tf.variable_scope('conv_block_3'):

					g_conv1_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv1_embedding = tf.contrib.layers.batch_norm(g_conv1_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv1_embedding = leaky_relu(g_conv1_embedding, name = 'outputs')

				short_cut = tf.layers.conv2d(short_cut, nb_channels, [1, 1], strides = (1, 1), padding = 'SAME')

				g_conv1_embedding = tf.add(g_conv1_embedding, short_cut)

				g_conv1_embedding = max_pool(g_conv1_embedding, ksize = [1, 2, 2, 1], strides = (1, 2, 2, 1), padding = 'SAME')


			with tf.variable_scope('res_block_2'):
				short_cut = g_conv1_embedding
				nb_channels = self.layer_sizes[1]
				with tf.variable_scope('conv_block_1'):
					g_conv2_embedding = tf.layers.conv2d(g_conv1_embedding, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv2_embedding = tf.contrib.layers.batch_norm(g_conv2_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv2_embedding = leaky_relu(g_conv2_embedding, name = 'outputs')
				with tf.variable_scope('conv_block_2'):
					g_conv2_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv2_embedding = tf.contrib.layers.batch_norm(g_conv2_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv2_embedding = leaky_relu(g_conv2_embedding, name = 'outputs')

				with tf.variable_scope('conv_block_3'):

					g_conv2_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv2_embedding = tf.contrib.layers.batch_norm(g_conv2_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv2_embedding = leaky_relu(g_conv2_embedding, name = 'outputs')

				short_cut = tf.layers.conv2d(short_cut, nb_channels, [1, 1], strides = (1, 1), padding = 'SAME')

				g_conv2_embedding = tf.add(g_conv2_embedding, short_cut)

				g_conv2_embedding = max_pool(g_conv2_embedding, ksize = [1, 2, 2, 1], strides = (1, 2, 2, 1), padding = 'SAME')


		with tf.variable_scope('res_block_3'):
				short_cut = g_conv2_embedding
				nb_channels = self.layer_sizes[2]
				with tf.variable_scope('conv_block_1'):
					g_conv3_embedding = tf.layers.conv2d(g_conv2_embedding, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv3_embedding = tf.contrib.layers.batch_norm(g_conv3_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv3_embedding = leaky_relu(g_conv3_embedding, name = 'outputs')
				with tf.variable_scope('conv_block_2'):
					g_conv3_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv3_embedding = tf.contrib.layers.batch_norm(g_conv3_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv3_embedding = leaky_relu(g_conv3_embedding, name = 'outputs')

				with tf.variable_scope('conv_block_3'):

					g_conv3_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv3_embedding = tf.contrib.layers.batch_norm(g_conv3_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv3_embedding = leaky_relu(g_conv3_embedding, name = 'outputs')

				short_cut = tf.layers.conv2d(short_cut, nb_channels, [1, 1], strides = (1, 1), padding = 'SAME')

				g_conv3_embedding = tf.add(g_conv3_embedding, short_cut)

				g_conv3_embedding = max_pool(g_conv3_embedding, ksize = [1, 2, 2, 1], strides = (1, 2, 2, 1), padding = 'SAME')
		

		with tf.variable_scope('res_block_4'):
				short_cut = g_conv3_embedding
				nb_channels = self.layer_sizes[3]
				with tf.variable_scope('conv_block_1'):
					g_conv4_embedding = tf.layers.conv2d(g_conv3_embedding, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv4_embedding = tf.contrib.layers.batch_norm(g_conv4_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv4_embedding = leaky_relu(g_conv4_embedding, name = 'outputs')
				with tf.variable_scope('conv_block_2'):
					g_conv4_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv4_embedding = tf.contrib.layers.batch_norm(g_conv4_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv4_embedding = leaky_relu(g_conv4_embedding, name = 'outputs')

				with tf.variable_scope('conv_block_3'):

					g_conv4_embedding = tf.layers.conv2d(image_input, nb_channels, [3, 3], strides = (1, 1),
						padding = 'SAME')

					g_conv4_embedding = tf.contrib.layers.batch_norm(g_conv4_embedding, updates_collection = None, decay = 0.99, 
																		scale = True, center = True, is_training = training_flag)
					g_conv4_embedding = leaky_relu(g_conv4_embedding, name = 'outputs')

				short_cut = tf.layers.conv2d(short_cut, nb_channels, [1, 1], strides = (1, 1), padding = 'SAME')

				g_conv4_embedding = tf.add(g_conv4_embedding, short_cut)

				g_conv4_embedding = max_pool(g_conv4_embedding, ksize = [1, 2, 2, 1], strides = (1, 2, 2, 1), padding = 'SAME')		


		with tf.variable_scope('layer_5'):
			nb_channels = self.layer_sizes[4]
			g_conv5_embedding = tf.layers.conv2d(g_conv4_embedding, nb_channels, [1, 1], strides = (1, 1), padding = 'SAME')

			g_conv5_embedding = leaky_relu(g_conv5_embedding, name = 'outputs')

			g_conv5_embedding = max_pool(g_conv5_embedding, ksize = [1, 2, 2, 1], strides = (1, 6, 6, 1), padding = 'SAME')

			g_conv5_embedding = tf.layers.dropout(g_conv5_embedding, rate = 0.5, training = training_flag)


		with tf.variable_scope('layers_6'):
			nb_channels = self.layer_sizes[5]

			g_conv6_embedding = tf.layers.conv2d(g_conv5_embedding, nb_channels, [1, 1], strides = (1, 1), padding = 'SAME')

			g_conv6_embedding = tf.contrib.layers.batch_norm(g_conv6_embedding, updates_collection = None, decay = 0.99, 
																scale = True, center = True, is_training = training_flag)

			g_conv6_embedding = tf.layers.dropout(g_conv6_embedding, rate = 0.5, training = training_flag)


	return g_conv6_embedding



class MatchingNetwork:
	def __init__(self, support_set_images, support_set_labels, target_image, target_label, keep_prob, 
					batch_size = 100, num_channels = 3, is_training = False, learning_rate = 0.001, fce = False, 
					ways = 5, shots = 1)

	self.batch_size = batch_size
	self.fce = fce
	self.g = EmbeddingNetwork(self.batch_size, num_channels = num_channels, layer_sizes = [64,96,128,256,2048,512])

	if fce:
		self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size = self.batch_size)

	self.dn = DistanceNetwork()
	self.classifiy = AttentionalClassify()
	self.support_set_images = support_set_images
	self.support_set_labels = support_set_labels
	self.target_image = target_image
	self.target_label = target_label
	self.keep_prob = keep_prob
	self.is_training = is_training
	self.ways = ways
	self.shots = shots
	self.learning_rate = learning_rate




	def losses():
		"""
		builds tf graph for mactching network, produces losses and summary statistics
		"""
		with tf.variable_scope("losses"):
			[b, ways, shots] = self.support_set_labels.get_shape().as_list()
			self.support_set_labels = tf.reshape(self.support_set_labels, shape = [b, ways*shots])
			self.support_set_labels = tf.one_hot(self.support_set_labels, self.ways)

			embedded_images = []

			[b, ways, shots, h, w, c] = self.support_set_images.get_shape().as_list()
			self.support_set_images = tf.reshape(self.support_set_images, shape = [b, ways*shots, h, w, c])
			for image in tf.unstack(self.support_set_images, axis = 1):
				gen_embeddings = self.g(image)





