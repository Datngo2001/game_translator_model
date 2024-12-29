import tensorflow as tf
mnist = tf.keras.datasets.mnist

dataset = mnist.load_dataset("yhavinga/ccmatrix", "en-nl", streaming=True)
