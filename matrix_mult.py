import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

# Launch the default graph.
with tf.Session() as sess:
# specify GPU
#    with tf.device("/gpu:0"):

#For cluster
#with tf.Session("grpc://example.org:2222") as sess:

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the op is returned in 'result' as a numpy `ndarray` object.
    result = sess.run(product)
    print(result) # ==> [[ 12.]]
# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")
one = tf.constant(1)
input = tf.constant([2.0])
#create operation
new_value = tf.add(state,one)
update = tf.assign(state, new_value)
#init
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    # Run operation that updates the state
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
    print sess.run([tf.mul(state,input),update])
