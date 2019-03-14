from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import IPython.display as display



# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def __test():
    print(_bytes_feature(b'test_string'))
    print(_bytes_feature(u'test_bytes'.encode('utf-8')))

    print(_float_feature(np.exp(1)))

    print(_int64_feature(True))
    print(_int64_feature(1))

    feature = _float_feature(np.exp(1))

    feature.SerializeToString()

def __test():
    # the number of observations in the dataset
    n_observations = int(1e4)

    # boolean feature, encoded as False or True
    feature0 = np.random.choice([False, True], n_observations)

    # integer feature, random from 0 .. 4
    feature1 = np.random.randint(0, 5, n_observations)

    # string feature
    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]

    # float feature, from a standard normal distribution
    feature3 = np.random.randn(n_observations)


def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }
  
  # Create a Features message using tf.train.Example.
  
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def __test():
    example_observation = []

    serialized_example = serialize_example(False, 4, b'goat', 0.9876)
    serialized_example


    example_proto = tf.train.Example.FromString(serialized_example)
    example_proto
    # TODO how to get the data back?

def __test():
    tf.data.Dataset.from_tensor_slices(feature1)
    features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
    features_dataset
    for f0,f1,f2,f3 in features_dataset.take(1):
      print(f0)
      print(f1)
      print(f2)
      print(f3)

def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_func(
    serialize_example, 
    (f0,f1,f2,f3),  # pass these args to the above function.
    tf.string)      # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.
  return tf.reshape(tf_string, ()) # The result is a scalar

def __test():
    serialized_features_dataset = features_dataset.map(tf_serialize_example)
    serialized_features_dataset
    filename = 'test.tfrecord'
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

def __test():
    # Reading from file
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    raw_dataset

    for raw_record in raw_dataset.take(10):
      print(repr(raw_record))

    # Create a description of the features.  
    feature_description = {
        'feature0': tf.FixedLenFeature([], tf.int64, default_value=0),
        'feature1': tf.FixedLenFeature([], tf.int64, default_value=0),
        'feature2': tf.FixedLenFeature([], tf.string, default_value=''),
        'feature3': tf.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)
    parsed_dataset

    for parsed_record in parsed_dataset.take(10):
      print(repr(parsed_record))

def convert_to_proto(data):
    # data format example: ('0x87', ['hello', 'world'], [1.2, 0.8])
    feature = {
        'id': _bytes_feature(data[0].encode('utf-8')),
        'mutate': _bytes_feature(tf.serialize_tensor(data[1]).numpy()),
        'label': _bytes_feature(tf.serialize_tensor(data[2]).numpy())
    }
    exp = tf.train.Example(features=tf.train.Features(feature=feature))
    return exp

def decode(raw):
    feature_description = {
        'id': tf.FixedLenFeature([], tf.string, default_value=''),
        'mutate': tf.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.FixedLenFeature([], tf.string, default_value='')
    }
    def _my_parse_function(pto):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.parse_single_example(pto, feature_description)
    decoded = raw.map(_my_parse_function)
    # need to further parse tensor
    # FIXME can we define this inside protobuffer?
    def _my_parse2(pto):
        return {'id': pto['id'],
                'label': tf.parse_tensor(pto['label'], out_type=tf.float32),
                'mutate': tf.parse_tensor(pto['mutate'], out_type=tf.string)}
    return decoded.map(_my_parse2)
    
def __test():
    exp1 = convert_to_proto(('0x87', ['hello', 'world'], [1.2, 0.8]))
    exp2 = convert_to_proto(('0x333', ['yes', 'and', 'no'], [-1, 0.33]))
    exp1
    with tf.python_io.TFRecordWriter('111.tfrec') as writer:
        writer.write(exp1.SerializeToString())
        writer.write(exp2.SerializeToString())
    raw = tf.data.TFRecordDataset('111.tfrec')
    out = decode(raw)
    out
    for a in out:
        print(a)

    

def mywrite():
    # some data
    data = ('0x87', ['hello', 'world'], [1.2, 0.8])
    tf.serialize_tensor(data[0].encode('utf8'))
    _bytes_feature(data[0])
    tf.parse_tensor(tf.serialize_tensor(data[1]).numpy(), out_type=tf.string)
    tf.parse_tensor(tf.serialize_tensor(data[2]).numpy(), out_type=tf.float32)
    sess = tf.Session()
    tf.serialize_tensor(data[1]).numpy()
    _bytes_feature(tf.serialize_tensor(data[1]).numpy())
    tf.train.FeatureList(bytes_list=tf.train.BytesList(value=[tf.serialize_tensor(data[1])]))
    s = exp.SerializeToString()
    exp = tf.train.Example.FromString(s)
    feature_description = {
        'id': tf.FixedLenFeature([], tf.string, default_value=''),
        'mutate': tf.FixedLenFeature([], tf.string, default_value=''),
        'label': tf.FixedLenFeature([], tf.string, default_value=''),
        # 'feature3': tf.FixedLenFeature([], tf.float32, default_value=0.0),
    }
    tf.parse_single_example(exp[0], feature_description)
    tf.parse_single_example(tf.data.Dataset.from_tensor_slices(exp), feature_description)

    with tf.python_io.TFRecordWriter('111.tfrec') as writer:
        writer.write(exp.SerializeToString())

    def _my_parse_function(pto):
      # Parse the input tf.Example proto using the dictionary above.
      return tf.parse_single_example(pto, feature_description)
    raw = tf.data.TFRecordDataset('111.tfrec')
    raw
    parsed_image_dataset = raw.map(_my_parse_function)
    parsed_image_dataset
    
    for a in parsed_image_dataset:
        print(a)
        print(a['label'])
        # shit!
        print(tf.parse_tensor(a['label'], out_type=tf.float32))
        
    feature['label']
    tf.parse_tensor(feature['label'], out_type=tf.float32)



