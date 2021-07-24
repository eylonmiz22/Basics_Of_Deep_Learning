import numpy as np
import argparse
import tensorflow as tf



def mse(p, gt):
    return tf.reduce_sum(tf.square(p-gt))/len(gt)


def linear(x, w, b):
    return tf.matmul(tf.cast(x,tf.float32), w) + b


def btu(x, w, b, temp=1e-3):
    return tf.sigmoid(linear(x,w,b) / temp)


def model(x, k, W, gt, is_bypass=False):
    
    out_channel = 1
    dim = len(x[0]) # Change
    (w1,b1), (w2,b2) = W

    # Assign weights and biases No'1
    w1_var = tf.Variable(tf.random.uniform([dim,k], -1, 1, seed=0))
    b1_var = tf.Variable(tf.zeros([k]))
    w1_var = tf.compat.v1.assign(w1_var, w1)
    b1_var = tf.compat.v1.assign(b1_var, b1)

    # Assign weights and biases No'2
    w2_var = tf.Variable(tf.random.uniform([dim+k, out_channel], -1, 1,
        seed=0)) if is_bypass else tf.Variable(tf.random.uniform([k,out_channel], -1, 1, seed=0))
    b2_var = tf.Variable(tf.zeros([out_channel]))
    w2_var = tf.compat.v1.assign(w2_var, w2)
    b2_var = tf.compat.v1.assign(b2_var, b2)

    hlayer = btu(x, w1_var, b1_var)
    if is_bypass:
        hlayer = tf.concat([hlayer,x], axis=1)

    out = btu(hlayer, w2_var, b2_var)
    return mse(out, gt), out
      

def result_to_file(txt_file, data, weights, mse_result, xor_result):
    s = "Weights and Biases:\n"
    for i, (w,b) in enumerate(weights):
        s += f"{i+1}. W{i+1} = {w}, B{i+1} = {b}\n"
    s += f"\nMSE Error: {mse_result}\n\nXOR Truth Table:\n\n|    A    |    B    | XOR |\n"
    for (a,b), xor in zip(data, xor_result):
        s += f"|    {a}    |    {b}    |    {int(xor[0])}    |\n"
    s += "\n"
    txt_file.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XOR TF implementation")
    parser.add_argument('-o', "--out_path", type=str, help="Output text file path")
    out_path = parser.parse_args().out_path

    with open(out_path, 'w') as results:

        # ************* k = 4 *************

        x = [[0,0], [0,1], [1,0], [1,1]]

        w1 = [[-1,-1,1,1], [-1,1,-1,1]]
        b1 = [0.5,-0.5,-0.5,-1.5]

        w2 = [[0],[1],[1],[0]]
        b2 = [-0.5]

        W = [(w1,b1), (w2,b2)]
        result_to_file(results, x, W, *model(x, 4, W, [[0],[1],[1],[0]]))


        # ************* k = 2 *************

        x = [[0, 0], [0, 1], [1, 0], [1, 1]]

        w1 = [[-1, 1], [1, -1]]
        b1 = [-0.5, -0.5]

        w2 = [[1], [1]]
        b2 = [-0.5]

        W = [(w1,b1), (w2,b2)]
        result_to_file(results, x, W, *model(x, 2, W, [[0],[1],[1],[0]]))


        # ************* k = 1 *************

        x = [[0, 0], [0, 1], [1, 0], [1, 1]]

        w1 = [[-1], [1]]
        b1 = [-0.5]

        w2 = [[2], [1], [-1]]
        b2 = [-0.5]

        W = [(w1,b1), (w2,b2)]
        result_to_file(results, x, W, *model(x, 1, W, [[0],[1],[1],[0]], is_bypass=True))
