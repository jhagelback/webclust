
/** ------------------------------------------------------

Implementation of K-Means clustering algorithm.

--------------------------------------------------------- */

class KMeans {
    constructor(k, distf, initf, seed) {
        this.k = k;
        this.data = [];
        this.centroids = [];
        global_seed = seed;
        
        // Distance function
        if (distf == "0") {
            this.dist_func = this.dist_euclidean;
        }
        else if(distf == "1") {
            this.dist_func = this.dist_manhattan;
        }
        else if(distf == "2") {
            this.dist_func = this.dist_chebyshev;
        }
        else {
            throw("Unknown distance function: " + dist);
        }

        // Initalizer function
        if (initf == "0") {
            this.init_func = this.random_init;
        }
        else if(initf == "1") {
            this.init_func = this.kmplusplus_init;
        }
        else {
            throw("Unknown initalizer function: " + initf);
        }

        // Current training iteration
        this.current_iter = 0;
        this.max_iter = 100;
        // Training done
        this.training_done = false;
    }

    // Returns number of clusters
    no_clusters() {
        return this.k;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return true;
    }
  
    // Random centroids initialization
    random_init(data) {
        for (let i = 0; i < this.k; i++) {
            let c = new KMCentroid(i + 1);
            c.set_random(data);
            this.centroids.push(c);
        }
    }

    // Calculates min squared distance between this instance and current centroids
    min_sqdist_centroids(inst) {
        // Iterate over all centroids
        for (let c = 0; c < this.centroids.length; c++) {
            // Calculate squared dist and see if we have a new min
            let sqdist = Math.pow(this.dist_func(inst, this.centroids[c]), 2);
            if (sqdist < inst.dist) {
                inst.centroid = c;
                inst.dist = sqdist;
            }
        }
        return inst.dist;
    }

    // K-Means++ initialization
    // See: http://www.real-statistics.com/multivariate-statistics/cluster-analysis/initializing-clusters-k-means/
    kmplusplus_init(data) {
        // Step 1: select one random instance as centroid
        let i = parseInt(rnd() * data.no_examples());
        let c = new KMCentroid(this.centroids.length + 1);
        c.set(data.x[i]);
        this.centroids.push(c);

        // Step 2: select other random instances as centroids based on how far
        // away they are from current centroids
        while (this.centroids.length < this.k) {
            // Calculate sum of min squared dists to current centroids
            let sum_dist = 0;
            for (let i = 0; i < data.no_examples(); i++) {
                let inst = new KMInstance(data.x[i]);
                sum_dist += this.min_sqdist_centroids(inst);
            }

            // Select a new instance to be centroid with probability
            // D(xi)^2 / Sum (D(xj)^2)
            let selected = false;
            while (!selected) {
                // Find random instance
                let i = parseInt(rnd() * data.no_examples());
                let inst = new KMInstance(data.x[i]);
                // Calculate probability
                let prob = this.min_sqdist_centroids(inst) / sum_dist;
                // If random value is equal or below probability, select this instance as centroid
                if (rnd() <= prob) {
                    let c = new KMCentroid(this.centroids.length + 1);
                    c.set(inst.x);
                    this.centroids.push(c);
                    selected = true;
                }
            }
        }
    }

    // Trains the clustering algorithm
    train(data) {
        this.init_func(data);
        
        // Add data to internal array
        for (let i = 0; i < data.no_examples(); i++) {
            let e = new KMInstance(data.x[i]);
            this.data.push(e);
        }
    }

    // Checks if the training is done or not
    done() {
        return this.training_done;
    }
    
    // Returns the current iteration
    current_iteration() {
        return this.current_iter;
    }
    
    // Executes all training iterations
    iterate_all() {
        while(!this.training_done) {
            this.iterate();
        }
    }
    
    // Executes one training iteration
    iterate() {
        // Flag if assignments have been changed or not
        let changed = false;

        // Assign each instance to the closest centroid
        for (let i = 0; i < this.data.length; i++) {
            let inst = this.data[i];
            inst.reset();

            // Calculate distance to each centroid
            for (let c = 0; c < this.centroids.length; c++) {
                let dist = this.dist_func(inst, this.centroids[c]);
                if (dist < inst.dist) {
                    inst.centroid = c;
                    inst.dist = dist;
                }
            }

            // Check if assignment for this instance has changed
            if (inst.centroid != inst.prev_centroid) {
                changed = true;
            }
        }

        // Move each centroid to mean
        for (let c = 0; c < this.centroids.length; c++) {
            let centroid = this.centroids[c];

            let new_x = new Array(centroid.no_attr()).fill(0);

            let cnt = 0;
            for (let i = 0; i < this.data.length; i++) {
                let inst = this.data[i];
                if (inst.centroid == c) {
                    cnt++;
                    for (let a = 0; a < centroid.no_attr(); a++) {
                        new_x[a] += inst.x[a];
                    }
                }
            }

            for (let a = 0; a < centroid.no_attr(); a++) {
                new_x[a] /= cnt;
            }
            centroid.x = new_x;
        }

        // Stop training if no new assignments were made
        if (!changed || this.current_iter >= this.max_iter) {
            this.training_done = true;    
        }
        this.current_iter++;
    }
    
    // Predicts a list of instances
    predict(instances) {
        let pred = [];
        for (let i = 0; i < instances.length; i++) {
            let inst = instances[i];
            pred.push(this.classify(inst));
        }
        return pred;
    }
    
    // Classifies which cluster an instance belongs to
    classify(i) {
        let inst = new KMInstance(i);
        inst.centroid = 0;
        inst.dist = this.dist_func(inst, this.centroids[0]);

        // Calculate distance to each centroid
        for (let c = 1; c < this.centroids.length; c++) {
            let dist = this.dist_func(inst, this.centroids[c]);
            if (dist < inst.dist) {
                inst.dist = dist;
                inst.centroid = c;
            }
        }
        
        return this.centroids[inst.centroid].label;
    }

    // Returns the markers to be rendered, in this case
    // the centroids
    get_markers() {
        let m = [];
        for (let c = 0; c < this.centroids.length; c++) {
            m.push(this.centroids[c].x);
        }
        return m;
    }
    
    // Calculates distance (Euclidean) between an instance and a centroid
    dist_euclidean(inst, c) {
        let sumSq = 0;
        for (let i = 0; i < c.no_attr(); i++) {
            sumSq += Math.pow(inst.x[i] - c.x[i], 2);
        }
        sumSq = Math.sqrt(sumSq);
        return sumSq;
    }

    // Calculates Manhattan distance between an instance and a centroid
    dist_manhattan(inst, c) {
        let sum = 0;
        for (let i = 0; i < c.no_attr(); i++) {
            sum += Math.abs(inst.x[i] - c.x[i]);
        }
        return sum;
    }

    // Calculates Chebyshev distance between an instance and a centroid
    dist_chebyshev(inst, c) {
        let best_v = 0;
        for (let i = 0; i < c.no_attr(); i++) {
            let v = Math.abs(inst.x[i] - c.x[i]);
            if (v > best_v) {
                best_v = v;
            }
        }
        return best_v;
    }
}

/**
    Holds a dataset instance.
*/
class KMInstance {
    constructor(x) {
        this.x = x;
        this.prev_centroid = -1;
        this.centroid = 0;
        this.dist = 100000;
    }

    // Reset assignment for this instance
    reset() {
        this.prev_centroid = this.centroid;
        this.centroid = 0;
        this.dist = 100000;
    }
}

/**
    Centroid class for K-Means.
*/
class KMCentroid {
    // Constructor
    constructor(label) {
        this.label = label;
    }

    // Set attribute values
    set(x) {
        this.x = x;
    }

    // Random initialization
    set_random(data) {
        // Create empty x array
        this.x = new Array(data.no_attr()).fill(0);;

        // Randomize all attributes between min and max
        for (let c = 0; c < data.no_attr(); c++) {
            let mi = data.attr_min(c);
            let ma = data.attr_max(c);
            let r = ma - mi;
            let rndval = mi + rnd() * r;
            this.x[c] = rndval;
        }
    }

    // Number of attributes
    no_attr() {
        return this.x.length;
    }
}
