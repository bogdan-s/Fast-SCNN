import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

ds = tf.data.Dataset.range(10)  

print(tf.data.experimental.cardinality(ds)) 

shuffled_and_batched = ds.cache().shuffle(62, reshuffle_each_iteration=True).map(lambda x: x + 100 * tf.random.uniform(shape = [], minval=1, maxval=5, dtype=tf.dtypes.int64, seed = 1)).batch(3).repeat(2)

# print(shuffled_and_batched.as_numpy_iterator())


for x in shuffled_and_batched.as_numpy_iterator():
    print(x)