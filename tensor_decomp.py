import numpy as np
import tensorflow as tf
import time

class CPDecomp(object):
    def __init__(self, shape, rank, sess, ndims=3, optimizer_type='adam'):
        '''
        `rank` is R, the number of 1D tensors to hold to get an approximation to `X`
        `optimizer_type` must be in ('adam', 'sgd')
        
        Approximates a tensor whose approximations are repeatedly fed in batch format to `self.train`
        '''
        self.rank = rank
        self.optimizer_type = optimizer_type
        self.shape = shape
        self.ndims = ndims
        self.sess = sess

        # t-th batch tensor
        # contains all data for this minibatch. already summed/averaged/whatever it needs to be. 
        self.X_t = tf.sparse_placeholder(tf.float32, shape=np.array(shape, dtype=np.int64))
        # Goal: X_ijk == sum_{r=1}^{R} U_{ir} V_{jr} W_{kr}
        self.U = tf.Variable(tf.random_uniform(
            shape=[self.shape[0], self.rank],
            minval=-1.0,
            maxval=1.0,
        ), name="U")
        self.V = tf.Variable(tf.random_uniform(
            shape=[self.shape[1], self.rank],
            minval=-1.0,
            maxval=1.0,
        ), name="V")
        if self.ndims > 2:
            self.W = tf.Variable(tf.random_uniform(
                shape=[self.shape[2], self.rank],
                minval=-1.0,
                maxval=1.0,
            ), name="W")
        self.create_loss_fn(reg_param=1e-8)

    def evaluate(self, X):
        '''
        `X` is the actual representation of `X_hat`. We return the RMSE between X_hat and X
        returns sqrt(1/#entries * sum_{entry} (X_{entry} - X_hat_{entry})^2

        WARNING: could use a lotta memory
        '''

        # X_hat ~= U x1 V x2 W - right?
        X_hat = tf.einsum('ir,jr,kr->ijk', self.U, self.V, self.W)
        tf_X = tf.constant(X, dtype=tf.float32)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X_hat, tf_X))))
        rmse_val = rmse.eval()
        print("RMSE: {}".format(rmse_val))
        
    def train_step(self, approx_tensor, print_every=10):
        if not hasattr(self, 'prev_time'):
            self.prev_time = time.time()
        feed_dict = {
            self.X_t: approx_tensor,
        }
        _, loss, step = self.sess.run(
            [
                self.train_op,
                self.loss,
                self.global_step,
            ],
            feed_dict=feed_dict,
        )

        if step % print_every == 0:
            print("Loss at step {}: {} (avg time per batch: {})".format(step, loss, (time.time() - self.prev_time) / print_every))
            self.prev_time = time.time()
        
    def create_loss_fn(self, reg_param):
        """
        L(X; U,V,W) = .5 sum_{i,j,k where X_ijk =/= 0} (X_ijk - sum_{r=1}^{R} U_ir V_jr W_kr)^2
        L_{rho} = L(X; U,V,W) + rho * (||U||^2 + ||V||^2 + ||W||^2) where ||.|| represents some norm (L2, L1, Frobenius)
        """
        def L(X, U,V,W):
            """
            X is a sparse tensor. U,V,W are dense. 
            """
            X_ijks = X.values  # of shape (N,) - represents all the values stored in X. 
            indices = tf.transpose(X.indices)  # of shape (N,3) - represents the indices of all values (in the same order as X.values)

            U_indices = tf.gather(indices, 0)  # of shape (N,) - represents all the indices to get from the U matrix
            V_indices = tf.gather(indices, 1)
            W_indices = tf.gather(indices, 2) # there better be an error!
            U_vects = tf.gather(U, U_indices)  # of shape (N, R) - each index represents the 1xR vector found in U_i
            V_vects = tf.gather(V, V_indices)
            W_vects = tf.gather(W, W_indices)

            # elementwise multiplication of each of U, V, and W - the first step in getting <U_i, V_j, W_k>, as a triple dot product (for each i,j,k in X)
            # we are calculating the matrix UVW (of shape N,R), where UVW_(m,:) = U_ir * V_jr * W_kr, where X.indices[m] = i,j,k.
            elementwise_product = tf.multiply(tf.multiply(U_vects, V_vects), W_vects)  # of shape (N, R)
                                                                                
            predicted_X_ijks = tf.reduce_sum(elementwise_product, axis=1)  # of shape (N,) - represents Sum_{r=1}^R U_ir V_jr W_kr
            errors = tf.square(X_ijks - predicted_X_ijks)  # of shape (N,) - elementwise error for each entry in X_ijk

            mean_loss = .5 * tf.reduce_sum(errors)  # average loss per entry in X - scalar!

            return mean_loss

        def reg(U,V,W):
            # NOTE: l2_loss already squares the norms. So we don't need to square them.
            summed_norms = (
                tf.nn.l2_loss(U, name="U_norm") +
                tf.nn.l2_loss(V, name="V_norm") +
                tf.nn.l2_loss(W, name="W_norm")
            )
            return (.5 * reg_param) * summed_norms

        self.loss = L(self.X_t, self.U,self.V,self.W) + reg(self.U, self.V, self.W)
        
    def get_train_op(self):
        # TODO: implement SALS or 2SGD. Also experiment with just using ADAM/SGD (builtin) to minimize the loss
        if self.optimizer_type == '2sgd':
            return self.get_train_op_2sgd()
        elif self.optimizer_type == 'sals':
            return self.get_train_op_sals()
        elif self.optimizer_type == 'adam':
            return self.get_train_op_adam()
        elif self.optimizer_type == 'sgd':
            return self.get_train_op_sgd()

    def get_train_op_2sgd(self, rho=1e-4):
        '''
        See 2SGD algorithm in Expected Tensor Decomp paper
        '''
        X = self.X_t
        U = self.U
        V = self.V
        W = self.W
        t = self.global_step
        eta_t = 1. / (1. + t)

        # X(.,V,W)_ir = sum_{j,k} X_ijk * V_jr * W_kr
        import pdb; pdb.set_trace()
        modified_X = tf.einsum('ijk,jr,kr->ir', X, V, W)

        def contract_X(X, V, W):
            result = tf.Variable(tf.zeros(shape=[self.shape[0], self.rank]), name='XVW')
            # TODO: generalize to X(U.W) and X(UV.)
            values = X.values
            indices = tf.transpose(X.indices)

            i_s = tf.gather(indices, 0)
            j_s = tf.gather(indices, 1)
            k_s = tf.gather(indices, 2)
            # TODO: do it out :)
            # for each (value, index) pair, add to result_ir
            for value, index in values, indices:
                result[index.i, :] += value * tf.multiply(v[index.j], w[index.k])

        def gamma(A,B):
            ATA = tf.matmul(A,A, transpose_a=True)  # A^T * A
            BTB = tf.matmul(B,B, transpose_a=True)  # B^T * B
            return tf.multiply(ATA, BTB)  # hadamard product of A^T*A and B^T*B

        gamma_rho = gamma(V,W) + rho * tf.eye(self.rank)
        inv_gamma_rho = tf.matrix_inverse(gamma_rho)
        grad_value = tf.matmul(modified_X, inv_gamma_rho)
        tf.assign(U, (1-eta_t) * U + eta_t * grad_value)

    def get_train_op_sals(self):
        pass

    def get_train_op_adam(self):
        return self.optimizer.minimize(self.loss, global_step=self.global_step)

    def get_train_op_sgd(self):
        return self.optimizer.minimize(self.loss, global_step=self.global_step)

    def train(self, expected_tensors, true_X=None):
        '''
        Assumes `expected_tensors` is a generator of sparse tensor values. 
        '''
        self.batch_num = 0
        with tf.device('/gpu:0'):
            self.global_step = tf.Variable(0.0, name='global_step', trainable=False)
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            elif self.optimizer_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)

            self.train_op = self.get_train_op()

            self.sess.run(tf.initialize_all_variables())
            with self.sess.as_default():
                for expected_tensor in expected_tensors:
                    if self.batch_num % 100 == 0 and true_X is not None:
                        self.evaluate(true_X)
                    self.train_step(expected_tensor)
                    self.batch_num += 1


def test_decomp():
    shape = [30, 40, 50]
    true_X = np.random.rand(30, 40, 50)
    batch_tensors = (true_X + (np.random.rand(30,40,50) - 0.5) for _ in range(10000))
    def sparse_batch_tensor_generator():
        for X_t in batch_tensors:
            idx = tf.where(tf.not_equal(X_t, 0.0))
            yield tf.SparseTensorValue(idx.eval(), tf.gather_nd(X_t, idx).eval(), X_t.shape)

    config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    sess = tf.Session(config=config)
    with sess.as_default():
        decomp_method = CPDecomp(
            shape=shape,
            sess=sess,
            rank=50,
            ndims=3,
            optimizer_type='adam',
        )
        print('training!')
        decomp_method.train(sparse_batch_tensor_generator(), true_X)


if __name__ == '__main__':
    print('testing CP decomp...')
    test_decomp()

