from src.imports import tf, K

class Capsule(tf.keras.layers.Layer):
    def __init__(self, num_clases=5, vec=16):
        super(Capsule, self).__init__()
        self.num_clases = num_clases
        self.vec = vec
        self.W = self.add_weight(
            shape=[1, 1, self.num_clases, self.vec, 512],
            initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),
            trainable=True,
            name='weights_matrix'
        )

    def call(self, inputs):
        primary_capsule_output = self.primary_capsule(inputs)
        routing_output = self.routing(primary_capsule_output)
        return routing_output
    
    def primary_capsule(self, inputs):
        u = tf.reshape(inputs, (-1, 1, 512))
        u = tf.expand_dims(u, axis=-2)
        u = tf.expand_dims(u, axis=-1)
        u_hat = tf.matmul(self.W, u)
        u_hat = tf.squeeze(u_hat, [4])
        return u_hat

    def routing(self, inputs):
        b = tf.zeros(shape=[tf.shape(inputs)[0], 2, self.num_clases, 1])
        
        for i in range(2):
            c = tf.nn.softmax(b, axis=-2)
            s = tf.reduce_sum(c * inputs, axis=1, keepdims=True)
            v = self.squash(s)
            agreement = tf.squeeze(tf.matmul(tf.expand_dims(inputs, axis=-1),
                                             tf.expand_dims(v, axis=-1),
                                             transpose_a=True), [4])
            b += agreement
        return v

    @staticmethod
    def squash(inputs):
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        return (squared_norm / (1 + squared_norm)) * (inputs / tf.sqrt(squared_norm + K.epsilon()))

    @staticmethod
    def safe_norm(v, axis=-1, epsilon=1e-7):
        v_ = tf.reduce_sum(tf.square(v), axis=axis, keepdims=True)
        return tf.sqrt(v_ + epsilon)
    
    @staticmethod
    def output_layer(inputs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
