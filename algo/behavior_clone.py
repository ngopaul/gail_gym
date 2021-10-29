import tensorflow as tf


class BehavioralCloning:
    def __init__(self, Policy):
        self.Policy = Policy

        self.actions_expert = tf.compat.v1.placeholder(tf.int32, shape=[None], name='actions_expert')

        actions_vec = tf.one_hot(self.actions_expert, depth=self.Policy.act_probs.shape[1], dtype=tf.float32)

        loss = tf.reduce_sum(input_tensor=actions_vec * tf.math.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
        loss = - tf.reduce_mean(input_tensor=loss)
        tf.compat.v1.summary.scalar('loss/cross_entropy', loss)

        optimizer = tf.compat.v1.train.AdamOptimizer()
        self.train_op = optimizer.minimize(loss)

        self.merged = tf.compat.v1.summary.merge_all()

    def train(self, obs, actions):
        return tf.compat.v1.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                                      self.actions_expert: actions})

    def get_summary(self, obs, actions):
        return tf.compat.v1.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.actions_expert: actions})

