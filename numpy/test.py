#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def main():
    input = np.array([0.5, 2])
    # w1 = np.array([[1,2,3], [3,4,5]])
    w1 = np.array([[1,2,3], [5,4,3]])
    w2 = np.array([3,2,1])

    np.array([[1,2,3],[4,5,6]]).dot(np.array([[1],[2],[3]]))
    np.array([[1,2,3],[4,5,6]]).dot(np.array([1,2,3]))

    np.prod(np.array([1,2,3]), np.array([4,5,6]))
    np.prod([2,3])
    np.prod(np.array([2,3,8]))

    tf.prod([1,2,3])
    
    return (input.dot(w1) + 1).dot(w2)+1

if __name__ == '__test__':
    main()
    x = tf.constant([[1., 1.], [2., 2.]])
    tf.reduce_mean(x)  # 1.5
    tf.reduce_mean(x, 0)  # [1.5, 1.5]
    tf.reduce_mean(x, 1)  # [1.,  2.]

    
