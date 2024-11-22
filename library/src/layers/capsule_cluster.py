from src.imports import tf, K

class Capsule(tf.keras.layers.Layer):
    def __init__(self, num_clases=2, vec=16, kmeans_iters=10):
        super(Capsule, self).__init__()
        self.num_clases = num_clases
        self.vec = vec
        self.kmeans_iters = kmeans_iters
        self.W = self.add_weight(
            shape=[1, 1, self.num_clases, self.vec, 512], # [1, 1,2 ,16 ,512]
            initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05),  #henormal fail, #glorotnormal fail
            trainable=True,
            name='weights_matrix'
        )
        self.centroids = self.add_weight(shape=(self.num_clases, self.vec),
                                        initializer='random_normal',
                                        trainable=True,
                                        name='centroids'
        )

    def call(self, inputs):
        primary_capsule_output = self.primary_capsule(inputs)
        clustered_output = self.kmeans_clustering(primary_capsule_output)
        routing_output = self.routing(clustered_output)
        return routing_output

    def primary_capsule(self, inputs):
        u = tf.reshape(inputs, (-1, 1, tf.shape(inputs)[-1])) # [-1, 1, 512]
        u = tf.expand_dims(u, axis=-2) # [-1, 1, 1, 512]
        u = tf.expand_dims(u, axis=-1) # [-1, 1, 512, 1]
        u_hat = tf.matmul(self.W, u) 
        u_hat = tf.squeeze(u_hat, [4]) 
        return u_hat # [-1, 1, num_clases, vec]

    def kmeans_clustering(self, inputs):
        """
        Realiza un K-means clustering usando TensorFlow.
        """
        inputs = tf.reshape(inputs, (-1, self.num_clases, 1, self.vec)) # [-1, num_clases, 1, vec]
           
        for _ in range(self.kmeans_iters):
            
            expanded_centroids = tf.reshape(self.centroids, (self.num_clases, 2, self.vec)) # [num_clases, 2, vec]
            distances = tf.reduce_sum(tf.square(inputs - expanded_centroids), axis=-1) # [-1, num_clases, 2]
            cluster_assignments = tf.argmin(distances, axis=2)
            
            new_centroids = []
            for cls in range(self.num_clases):
                    cluster_points = tf.boolean_mask(inputs[:, cls], tf.equal(cluster_assignments[:, cls]))
                    new_centroid = tf.cond(
                        tf.size(cluster_points) > 0,
                        lambda: tf.reduce_mean(cluster_points, axis=0),
                        lambda: tf.expand_dims(centroids[cls], axis=0)
                    )
                    new_centroid = tf.reshape(new_centroid, (1, self.vec))
                    new_centroids.append(new_centroid)
            centroids = tf.concat(new_centroids, axis=0)
        
        expanded_centroids = tf.reshape(centroids, (self.num_clases, 2, self.vec)) # [num_clases, 2, vec]
        distances = tf.reduce_sum(tf.square(inputs - expanded_centroids), axis=-1)
        cluster_assignments = tf.argmin(distances, axis=2)
        
        clustered_output = tf.gather(centroids, tf.cast(cluster_assignments, tf.int64) + tf.range(self.num_clases, dtype=tf.int64)*2)
        return clustered_output
    
    def routing(self, inputs):
        inputs = tf.expand_dims(inputs, axis=2)
        inputs = tf.reshape(inputs, (-1, self.num_clases, 1, self.vec))

        b = tf.zeros(shape=[tf.shape(inputs)[0], 2, self.num_clases, 1])
        
        for i in range(2):
            c = tf.nn.softmax(b, axis=-2)
            s = tf.reduce_sum(c * inputs, axis=2, keepdims=True)
            v = self.squash(s)
            agreement = tf.squeeze(tf.matmul(tf.expand_dims(inputs, axis=-1),
                                             tf.expand_dims(v, axis=-1),
                                             transpose_a=True), [4])
            b += agreement
        return v # [-1, num_clases, 1, vec]

    @staticmethod
    def squash(inputs, epsilon=1e-7):
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
        return (squared_norm / (1 + squared_norm)) * (inputs / tf.sqrt(squared_norm + epsilon))

    @staticmethod
    def safe_norm(v, axis=-1, epsilon=1e-7):
        v_ = tf.reduce_sum(tf.square(v), axis=axis, keepdims=True)
        return tf.sqrt(v_ + epsilon)

    @staticmethod
    def output_layer(inputs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())
    

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_clases": self.num_clases,
            "vec": self.vec,
            "kmeans_iters": self.kmeans_iters,
        })
        return config
