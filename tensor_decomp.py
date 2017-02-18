import datetime
import numpy as np
import os
import tensorflow as tf
import time

class CPDecomp(object):
    def __init__(self, shape, rank, sess, ndims=3, optimizer_type='2sgd', reg_param=1e-10):
        '''
        `rank` is R, the number of 1D tensors to hold to get an approximation to `X`
        `optimizer_type` must be in ('adam', 'sgd', 'sals', '2sgd')
        
        Approximates a tensor whose approximations are repeatedly fed in batch format to `self.train`
        '''
        self.rank = rank
        self.optimizer_type = optimizer_type
        self.shape = shape
        self.ndims = ndims
        self.sess = sess

        with tf.device('/cpu:0'):
            # t-th batch tensor
            # contains all data for this minibatch. already summed/averaged/whatever it needs to be. 
            #self.X_t = tf.sparse_placeholder(tf.float32, shape=np.array(shape, dtype=np.int64))
            self.indices = tf.placeholder(tf.int64, shape=[None, self.ndims], name='X_t_indices')
            self.values = tf.placeholder(tf.float32, shape=[None], name='X_t_values')
            shape_sparse = np.array(self.shape, dtype=np.int64)
            self.X_t = tf.SparseTensorValue(self.indices, self.values, shape=shape_sparse)
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
        self.create_loss_fn(reg_param=reg_param)

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
        if self.results_file is not None:
            if self.batch_num == 0:
                print("RMSE", file=self.results_file)
            print("{}".format(rmse_val), file=self.results_file)
        print("RMSE (step {}): {}".format(self.batch_num, rmse_val))
       
    def train_step(self, approx_indices, approx_values, print_every=5):
        if not hasattr(self, 'prev_time'):
            self.prev_time = time.time()
            self.avg_time = 0.0
            self.total_recordings = 0
        feed_dict = {
            self.indices: approx_indices,
            self.values: approx_values,
        }
        _, loss, loss_summary, step = self.sess.run(
            [
                self.train_ops, # might need multiple train ops to be executed sequentially (see the case of sals)
                self.loss,
                self.loss_summary,
                self.global_step,
            ],
            feed_dict=feed_dict,
        )
        if self.write_loss:
            self.train_summary_writer.add_summary(loss_summary, step)
        if self.checkpoint_every is not None:
            if step % self.checkpoint_every == 0 and step > 0:
                t = time.time()
                print('Saving checkpoint at step {}...'.format(step))
                path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=self.global_step)
                print('Saved model checkpoint to {} (it took {} secs)'.format(path, time.time() - t))

        if step % print_every == 0:
            batch_time = (time.time() - self.prev_time) / print_every
            print("Loss at step {}: {} (avg time per batch: {})".format(step, loss, batch_time))
            self.prev_time = time.time()
            self.avg_time = (batch_time + self.total_recordings * self.avg_time) / (self.total_recordings + 1.0)
            #print("avg time (per batch, overall): {}".format(self.avg_time))
            self.total_recordings += 1
            #print("self.U: ", self.U.eval())
        
    def create_loss_fn(self, reg_param):
        """
        L(X; U,V,W) = .5 sum_{i,j,k where X_ijk =/= 0} (X_ijk - sum_{r=1}^{R} U_ir V_jr W_kr)^2
        L_{rho} = L(X; U,V,W) + rho * (||U||^2 + ||V||^2 + ||W||^2) where ||.|| represents some norm (L2, L1, Frobenius)
        """
        def L(X, U,V,W):
            """
            X is a sparse tensor. U,V,W are dense. 
            """

            indices = tf.cast(X.indices, tf.int32)
            indices = tf.transpose(indices)  # of shape (N,3) - represents the indices of all values (in the same order as X.values)
            with tf.device('/gpu:1'):
                X_ijks = X.values  # of shape (N,) - represents all the values stored in X. 

                U_indices = tf.gather(indices, 0)  # of shape (N,) - represents all the indices to get from the U matrix
                V_indices = tf.gather(indices, 1)
                W_indices = tf.gather(indices, 2)
                U_vects = tf.gather(U, U_indices)  # of shape (N, R) - each index represents the 1xR vector found in U_i
                V_vects = tf.gather(V, V_indices)
                W_vects = tf.gather(W, W_indices)

                # elementwise multiplication of each of U, V, and W - the first step in getting <U_i, V_j, W_k>, as a triple dot product (for each i,j,k in X)
                # we are calculating the matrix UVW (of shape N,R), where UVW_(m,:) = U_ir * V_jr * W_kr, where X.indices[m] = i,j,k.
                elementwise_product = tf.multiply(tf.multiply(U_vects, V_vects), W_vects)  # of shape (N, R)
                                                                                    
                predicted_X_ijks = tf.reduce_sum(elementwise_product, axis=1)  # of shape (N,) - represents Sum_{r=1}^R U_ir V_jr W_kr
                errors = tf.square(X_ijks - predicted_X_ijks)  # of shape (N,) - elementwise error for each entry in X_ijk

                mean_loss = .5 * tf.reduce_mean(errors)  # average loss per entry in X - scalar!

                return mean_loss

        def reg(U,V,W):
            # NOTE: l2_loss already squares the norms. So we don't need to square them.
            with tf.device('/gpu:1'):
                summed_norms = (
                    tf.nn.l2_loss(U, name="U_norm") +
                    tf.nn.l2_loss(V, name="V_norm") +
                    tf.nn.l2_loss(W, name="W_norm")
                )
                return (.5 * reg_param) * summed_norms

        if reg_param > 0.0:
            self.loss = L(self.X_t, self.U,self.V,self.W) + reg(self.U, self.V, self.W)
        else:
            self.loss = L(self.X_t, self.U,self.V,self.W)
        
    def get_train_ops(self):
        if self.optimizer_type == '2sgd':
            train_ops = [self.get_train_op_2sgd()]
        elif self.optimizer_type == 'sals':
            train_ops = self.get_train_ops_sals()
        elif self.optimizer_type == 'adam':
            train_ops = [self.get_train_op_adam()]
        elif self.optimizer_type == 'sgd':
            train_ops = [self.get_train_op_sgd()]
        inc_t = tf.assign(self.global_step, self.global_step+1)
        return [*train_ops, inc_t]

    def get_update_UVW_ops_for_2sgd_sals(self, rho):
        '''
        See 2SGD/SALS algorithms in Expected Tensor Decomp paper
        '''
        def gamma(A,B):
            ATA = tf.matmul(A,A, transpose_a=True)  # A^T * A
            BTB = tf.matmul(B,B, transpose_a=True)  # B^T * B
            return tf.multiply(ATA, BTB)  # hadamard product of A^T*A and B^T*B

        X = self.X_t
        U = self.U
        V = self.V
        W = self.W
        t = self.global_step
        alpha = .5  # smaller => decays slower (more quickly get updates from the gradients)
        batch_size = 1. / 500. * tf.sqrt(tf.cast(X.shape[0], tf.float32))
        eta_t = batch_size / (1. + t**alpha)

        indices = tf.cast(X.indices, tf.int32)
        indices = tf.transpose(indices)

        i_s = tf.gather(indices, 0)
        j_s = tf.gather(indices, 1)
        k_s = tf.gather(indices, 2)

        def contract_X_U(X, V, W):
            '''
            X(.,V,W)_ir = sum_{j,k} X_ijk * V_jr * W_kr
            '''
            values = X.values

            # Can't broadcast a vector (1xN) and a NxM matrix - need a MxN to broadcast the 1xN against
            elemwise_mult = tf.transpose(tf.multiply(tf.transpose(tf.gather(V, j_s)), values))
            each_contracted_vector = tf.multiply(elemwise_mult, tf.gather(W, k_s))  # multiply a 1xN, RxN, and RxN matrices (elementwise) to get an RxN dimensional matrix
            index_mappings = []
            for i in range(self.shape[0]):  # TODO: maybe reconsider this. this is insanely unproductive and slow. 
                idx_i = tf.where(tf.equal(i_s, i))  # this is potentially n^2 (if tf implements `where` naively)
                contracted_vects_for_i = tf.gather(each_contracted_vector, idx_i)  # of shape (#nonzeroes, 1, R)
                index_mappings.append(tf.reduce_sum(contracted_vects_for_i, axis=[0,1]))  # remove the #nonzeroes and the 1
            # Now index_mappings is a length-I list of R-dimensional vectors
            result = tf.stack(index_mappings, name='X_VW')  # IxR!
            return result

        def contract_X_V(X, U, W):
            '''
            X(U,.,W)_jr = sum_{i,k} X_ijk * U_ir * W_kr
            '''
            values = X.values

            # Can't broadcast a vector (1xN) and a NxM matrix - need a MxN to broadcast the 1xN against
            elemwise_mult = tf.transpose(tf.multiply(tf.transpose(tf.gather(U, i_s)), values))
            each_contracted_vector = tf.multiply(elemwise_mult, tf.gather(W, k_s))  # multiply a 1xN, RxN, and RxN matrices (elementwise) to get an RxN dimensional matrix
            index_mappings = []
            for j in range(self.shape[1]):
                idx_j = tf.where(tf.equal(j_s, j))  # this is potentially n^2 (if tf implements `where` naively)
                contracted_vects_for_j = tf.gather(each_contracted_vector, idx_j)  # of shape (#nonzeroes, 1, R)
                index_mappings.append(tf.reduce_sum(contracted_vects_for_j, axis=[0,1]))  # remove the #nonzeroes and the 1
            # Now index_mappings is a length-J list of R-dimensional vectors
            result = tf.stack(index_mappings, name='XU_W')  # JxR!
            return result

        def contract_X_W(X, U, V):
            '''
            X(U,V,.)_kr = sum_{i,j} X_ijk * U_ir * V_jr
            '''
            values = X.values

            # Can't broadcast a vector (1xN) and a NxM matrix - need a MxN to broadcast the 1xN against
            elemwise_mult = tf.transpose(tf.multiply(tf.transpose(tf.gather(U, i_s)), values))
            each_contracted_vector = tf.multiply(elemwise_mult, tf.gather(V, j_s))  # multiply a 1xN, RxN, and RxN matrices (elementwise) to get an RxN dimensional matrix
            index_mappings = []
            for k in range(self.shape[2]):
                idx_k = tf.where(tf.equal(k_s, k))  # this is potentially n^2 (if tf implements `where` naively)
                contracted_vects_for_k = tf.gather(each_contracted_vector, idx_k)  # of shape (#nonzeroes, 1, 10)
                index_mappings.append(tf.reduce_sum(contracted_vects_for_k, axis=[0,1]))  # remove the #nonzeroes and the 1
            # Now index_mappings is a length-K list of R-dimensional vectors
            result = tf.stack(index_mappings, name='XUV_')  # KxR!
            return result

        modified_X = contract_X_U(self.X_t, self.V, self.W)
        gamma_rho = gamma(V,W) + rho * tf.eye(self.rank)
        self.gr1 = gamma_rho
        inv_gamma_rho = tf.matrix_inverse(gamma_rho)
        self.gr1inv = gamma_rho
        grad_value_U = tf.matmul(modified_X, inv_gamma_rho)

        modified_X = contract_X_V(self.X_t, self.U, self.W)
        gamma_rho = gamma(U,W) + rho * tf.eye(self.rank)
        self.gr2 = gamma_rho
        inv_gamma_rho = tf.matrix_inverse(gamma_rho)
        grad_value_V = tf.matmul(modified_X, inv_gamma_rho)

        modified_X = contract_X_W(self.X_t, self.U, self.V)
        gamma_rho = gamma(U,V) + rho * tf.eye(self.rank)
        self.gr3 = gamma_rho
        inv_gamma_rho = tf.matrix_inverse(gamma_rho)
        grad_value_W = tf.matmul(modified_X, inv_gamma_rho)

        update_U_op = tf.assign(U, (1-eta_t) * U + eta_t * grad_value_U)
        update_V_op = tf.assign(V, (1-eta_t) * V + eta_t * grad_value_V)
        update_W_op = tf.assign(W, (1-eta_t) * W + eta_t * grad_value_W)
        return [update_U_op, update_V_op, update_W_op]

    def get_train_op_2sgd(self, rho=1e-4):
        [update_U_op, update_V_op, update_W_op] = self.get_update_UVW_ops_for_2sgd_sals(rho)
        # Update U,V,W simultaneously - I believe tf.group does this?
        update_CP_op = tf.group(update_U_op, update_V_op, update_W_op)
        return update_CP_op

    def get_train_ops_sals(self, rho=1e-4):
        [update_U_op, update_V_op, update_W_op] = self.get_update_UVW_ops_for_2sgd_sals(rho)
        # update U,V,W in order
        return [update_U_op, update_V_op, update_W_op]

    def get_train_op_adam(self):
        return self.optimizer.minimize(self.loss)

    def get_train_op_sgd(self):
        return self.optimizer.minimize(self.loss)

    def train(self, expected_tensors, true_X=None, evaluate_every=100, results_file=None, write_loss=True, checkpoint_every=None):
        '''
        Assumes `expected_tensors` is a generator of sparse tensor values. 
        '''
        self.batch_num = 0
        self.results_file = results_file
        num_invalid_arg_exceptions = 0
        with tf.device('/gpu:0'):
            print('setting up variables...')
            self.global_step = tf.Variable(0.0, name='global_step', trainable=False)
            if self.optimizer_type == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            elif self.optimizer_type == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)

            self.train_ops = self.get_train_ops()

        self.write_loss = write_loss
        if self.write_loss:
            timestamp = str(datetime.datetime.now())
            out_dir = os.path.abspath(os.path.join(os.path.curdir, 'tf_logs', timestamp))
            print('Writing summaries to {}.'.format(out_dir))
            self.loss_summary = tf.summary.scalar('loss', self.loss)
            self.train_summary_writer = tf.summary.FileWriter(os.path.join(out_dir, 'summaries'), self.sess.graph)
        else:
            self.loss_summary = self.global_step  # so it doesnt crash when i try to evaluate loss_summary

        self.checkpoint_every = checkpoint_every
        if self.checkpoint_every is not None:
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            self.checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)

        print('initializing variables...')
        self.sess.run(tf.global_variables_initializer())
        with self.sess.as_default():
            print('looping through batches...')
            self.sess.graph.finalize()
            for expected_indices, expected_values in expected_tensors:
                if self.batch_num % evaluate_every == 0 and true_X is not None:
                    self.evaluate(true_X)
                try:
                    self.train_step(expected_indices, expected_values)
                except tf.errors.InvalidArgumentError as e:
                    self.batch_num -= 1
                    num_invalid_arg_exceptions += 1
                    print("INVALID ARG EXCEPTION: {}. Accidentally noninvertible matrix? There have been {} of these.".format(e, num_invalid_arg_exceptions))
                    import pdb; pdb.set_trace()
                self.batch_num += 1
            if hasattr(self, 'avg_time') and results_file is not None:
                print('avg batch time: {}'.format(self.avg_time), file=results_file)
        self.train_summary_writer.close()


def test_decomp():
    shape = [30, 40, 50]
    true_U = np.random.rand(shape[0], 5)
    true_V = np.random.rand(shape[1], 5)
    true_W = np.random.rand(shape[2], 5)
    true_X = np.einsum('ir,jr,kr->ijk', true_U, true_V, true_W)

    def batch_tensors_gen(n):
        for _ in range(n):
            yield true_X + (np.random.rand(*tuple(shape)) - 0.5) 

    def sparse_batch_tensor_generator(n=1500):
        import random
        with tf.device('/cpu:0'):
            for X_t in batch_tensors_gen(n):
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            if random.random() < 0.99:
                                X_t[i,j,k] = 0
                idx = tf.where(tf.not_equal(X_t, 0.0))
                indices = idx.eval().astype(np.uint16)
                values = tf.gather_nd(X_t, idx).eval()
                #print('{} nonzero vals'.format(len(indices)))
                yield (indices, values)
                #yield tf.SparseTensorValue(indices, values, X_t.shape)

    config = tf.ConfigProto(
        allow_soft_placement=True,
    )
    sess = tf.Session(config=config)
    print('training (on 2sgd)!')
    with open('results_2sgd.txt', 'w') as f:
        # train 2sgd
        decomp_method = CPDecomp(
            shape=shape,
            sess=sess,
            rank=300,
            ndims=3,
            optimizer_type='2sgd',
        )
        decomp_method.train(sparse_batch_tensor_generator(), true_X=None, evaluate_every=2, results_file=f, write_loss=False)


if __name__ == '__main__':
    print('testing CP decomp...')
    test_decomp()

