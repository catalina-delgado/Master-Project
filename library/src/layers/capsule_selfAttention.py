from src.imports import tf, K

class CapsuleSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_clases=2, vec=16, input_dim=512):
        super(CapsuleSelfAttention, self).__init__()
        self.num_clases = num_clases
        self.vec = vec
        self.input_dim = input_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=[1, 1, self.num_clases, self.vec, self.input_dim, ], # [2, 1, 512, ,16]
            initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),  #henormal fail, #glorotnormal fail
            trainable=True,
            name='weights_matrix'
        )
        self.b = self.add_weight(
            shape=[1, self.num_clases, self.num_clases,1], 
            initializer=tf.zeros_initializer(), 
            name='b'
        )
        #self.conv2D = tf.keras.layers.Conv2D(self.num_clases, kernel_size=3, strides=1, activation='linear', groups=self.num_clases, padding='same')

    def call(self, inputs):
        primary_capsule_output = self.primary_capsule(inputs)
        routing_output = self.routing(primary_capsule_output)
        return routing_output

    def primary_capsule(self, inputs):
        #x = self.conv2D(inputs)
        x = tf.reshape(inputs, (-1, 1, self.input_dim)) # [none, vec, dim]
        x = self.squash(x) 
        return x #[none, 1, input_dim]

    def routing(self, inputs):
        # SELF ATTENTION

        u = tf.expand_dims(inputs, axis=-2)
        u = tf.expand_dims(u, axis=-1)
        u = tf.matmul(self.W, u)  
        u = tf.squeeze(u, [4]) # [none, 1, 2, 16]
       
        u_flat = tf.reshape(u, [-1, self.num_clases, self.vec])  # (None, nun_clases, vec) 
        c = tf.matmul(u_flat, u_flat, transpose_b=True)  # Producto punto de u con sí mismo
        c = tf.expand_dims(c, -1) # (None, num_clases, num_clases, 1)

        # Normalización y softmax:
        c = c / tf.sqrt(tf.cast(self.vec, tf.float32))  
        c = tf.nn.softmax(c, axis=1) 
        c = c + self.b  # Sumar el sesgo

        s = tf.reduce_sum(c * u, axis=-2) 
        v = self.squash(s)  
        v = self.safe_norm(v) # [none, nun_clases, 1]

        return v 

    @staticmethod
    def squash(inputs, epsilon=1e-7):
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        return (squared_norm / (1 + squared_norm)) * (inputs / tf.sqrt(squared_norm + epsilon))


    @staticmethod
    def safe_norm(v, axis=-1, epsilon=1e-7):
        v_ = tf.reduce_sum(tf.square(v), axis=axis, keepdims=True)
        return tf.sqrt(v_ + epsilon)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_clases": self.num_clases,
            "vec": self.vec,
            'input_dim': self.input_dim
        })
        return config
