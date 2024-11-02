from src.imports import tf, K

class Capsule(tf.keras.layers.Layer):
    def __init__(self, num_clases=2, vec=16, n_clusters=2, kmeans_iters=10):
        super(Capsule, self).__init__()
        self.num_clases = num_clases
        self.vec = vec
        self.n_clusters = n_clusters
        self.kmeans_iters = kmeans_iters
        self.W = None

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=[1, 1, self.num_clases, self.vec, input_shape[-1]], #[1,1,2,16,512]
            initializer=tf.keras.initializers.HeUniform(),  
            trainable=True,
            name='weights_matrix'
        )
        super(Capsule, self).build(input_shape)

    def call(self, inputs):
        primary_capsule_output = self.primary_capsule(inputs)
        clustered_output = self.kmeans_clustering(primary_capsule_output)
        routing_output = self.routing(clustered_output)
        return routing_output

    def primary_capsule(self, inputs):
        u = tf.reshape(inputs, (-1, 1, inputs[-1])) #[-1,1,512]
        u = tf.expand_dims(u, axis=-2)
        u = tf.expand_dims(u, axis=-1)
        u_hat = tf.matmul(self.W, u)
        u_hat = tf.squeeze(u_hat, [4])
        return u_hat

    def kmeans_clustering(self, inputs):
        """
        Realiza un K-means clustering usando TensorFlow.
        """
        initial_centroids = tf.gather(inputs, tf.random.shuffle(tf.range(tf.shape(inputs)[0]))[:self.n_clusters])
        
        centroids = initial_centroids
        for _ in range(self.kmeans_iters):
            expanded_inputs = tf.expand_dims(inputs, 1)
            expanded_centroids = tf.expand_dims(centroids, 0)
            
            distances = tf.reduce_sum(tf.square(expanded_inputs - expanded_centroids), axis=-1)
            
            cluster_assignments = tf.argmin(distances, axis=1)
            
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = tf.boolean_mask(inputs, tf.equal(cluster_assignments, k))
                new_centroid = tf.reduce_mean(cluster_points, axis=0)
                new_centroids.append(new_centroid)
            
            centroids = tf.stack(new_centroids)
        
        expanded_centroids = tf.expand_dims(centroids, 0)
        expanded_inputs = tf.expand_dims(inputs, 1)
        distances = tf.reduce_sum(tf.square(expanded_inputs - expanded_centroids), axis=-1)
        cluster_assignments = tf.argmin(distances, axis=1)
        
        clustered_output = tf.gather(centroids, cluster_assignments)
        return clustered_output

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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_clases": self.num_clases,
            "vec": self.vec,
            "n_clusters": self.n_clusters,
            "kmeans_iters": self.kmeans_iters,
        })
        return config
