# embedding layer
with tf.name_scope("embedding"):
    self.W = tf.Variable(tf.random_uniform([self._config.vocab_size, self._config.embedding_dim], -1.0, 1.0),
                         name="W")
    self.char_emb = tf.nn.embedding_lookup(self.W, self.input_x)
    self.char_emb_expanded = tf.expand_dims(self.char_emb, -1)
    tf.logging.info("Shape of embedding_chars:{}".format(str(self.char_emb_expanded.shape)))
# convolution + pooling layer
pooled_outputs = []
for i, filter_size in enumerate(self._config.filter_sizes):
with tf.variable_scope("conv-maxpool-%s" % filter_size):
    # convolution layer
    filter_width = self._config.embedding_dim
    input_channel_num = 1
    output_channel_num = self._config.num_filters
    filter_shape = [filter_size, filter_width, input_channel_num, output_channel_num]

    n = filter_size * filter_width * input_channel_num
    kernal = tf.get_variable(name="kernal",
                             shape=filter_shape,
                             dtype=tf.float32,
                             initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    bias = tf.get_variable(name="bias",
                           shape=[output_channel_num],
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer)
    # apply convolution process
    # conv shape: [batch_size, max_seq_len - filter_size + 1, 1, output_channel_num]
    conv = tf.nn.conv2d(
        input=self.char_emb_expanded,
        filter=kernal,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="cov")
    tf.logging.info("Shape of Conv:{}".format(str(conv.shape)))

    # apply non-linerity
    h = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
    tf.logging.info("Shape of h:{}".format(str(h)))

    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(
        value=h,
        ksize=[1, self._config.max_seq_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="pool"
    )
    tf.logging.info("Shape of pooled:{}".format(str(pooled.shape)))
    pooled_outputs.append(pooled)
    tf.logging.info("Shape of pooled_outputs:{}".format(str(np.array(pooled_outputs).shape)))

# concatenate all filter's output
total_filter_num = self._config.num_filters * len(self._config.filter_sizes)
all_features = tf.reshape(tf.concat(pooled_outputs, axis=-1), [-1, total_filter_num])
tf.logging.info("Shape of all_features:{}".format(str(all_features.shape)))
with tf.name_scope("output"):
W = tf.get_variable(
    name="W",
    shape=[total_filter_num, self._config.label_size],
    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape=[self._config.label_size]), name="b")
l2_loss += tf.nn.l2_loss(W)
l2_loss += tf.nn.l2_loss(b)
self.scores = tf.nn.xw_plus_b(all_features, W, b, name="scores")
self.predictions = tf.argmax(self.scores, 1, name="predictions")

# compute loss
with tf.name_scope("loss"):
losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
self.loss = tf.reduce_mean(losses) + self._config.l2_reg_lambda * l2_loss
def train(x_train, y_train, vocab_processor, x_dev, y_dev, model_config):
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNNModel(
            config=model_config,
            is_training=FLAGS.is_train
        )
        # Define Training proceduce
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Checkpoint directory, Tensorflow assumes this directioon already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(FLAGS.output_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep_checkpoint_max)

        # Write vocabulary
        vocab_processor.save(os.path.join(FLAGS.output_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A singel training step
            :param x_batch:
            :param y_batch:
            :return:
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            tf.logging.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch
            }
            step, loss, accuracy = sess.run(
                [global_step, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            tf.logging.info("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        # Generate batches
        batches = data.DataSet.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop, For each batch ..
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % FLAGS.save_checkpoints_steps == 0:
                tf.logging.info("\nEvaluation:")
                dev_step(x_dev, y_dev)
            if current_step % FLAGS.save_checkpoints_steps == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.logging.info("Saved model checkpoint to {}\n".format(path))
class DataSet(object):
def __init__(self, positive_data_file, negative_data_file):
    self.x_text, self.y = self.load_data_and_labels(positive_data_file, negative_data_file)

def load_data_and_labels(self, positive_data_file, negative_data_file):
    # load data from files
    positive_data = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_data = [s.strip() for s in positive_data]
    negative_data = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_data = [s.strip() for s in negative_data]

    # split by words
    x_text = positive_data + negative_data
    x_text = [self.clean_str(sent) for sent in x_text]

    # generate labels
    positive_labels = [[0, 1] for _ in positive_data]
    negative_labels = [[1, 0] for _ in negative_data]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def clean_str(self, string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            # !/bin/bash
            export
            CUDA_VISIBLE_DEVICES = 0
            # 如果运行的话，更改code_dir目录
            CODE_DIR = "/home/work/work/modifyAI/textCNN"
            MODEL_DIR =$CODE_DIR / model
            TRAIN_DATA_DIR =$CODE_DIR / data_set
