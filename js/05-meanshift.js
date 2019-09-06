
/** ------------------------------------------------------

Implementation of Mean-Shift clustering algorithm.
See: https://pythonprogramming.net/mean-shift-from-scratch-python-machine-learning-tutorial/

--------------------------------------------------------- */

class MeanShift {
    constructor(bandwidth, distf, weightf) {
        this.data = [];
        this.centroids = [];
        this.bandwidth = bandwidth;
        this.weightf = weightf;
        global_seed = 42;
        
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

        // Current training iteration
        this.current_iter = 0;
        this.max_iter = 100;
        // Training done
        this.training_done = false;
    }

    // Returns number of clusters
    no_clusters() {
        return this.centroids.length;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return true;
    }
  
    // Init all data points as centroids
    init() {
        for (let i = 0; i < this.data.length; i++) {
            let inst = this.data[i];
            let c = new MSCentroid(inst.x);
            this.centroids.push(c);
        }
    }

    // Trains the clustering algorithm
    train(data) {    
        // Add data to internal array
        for (let i = 0; i < data.no_examples(); i++) {
            let e = new MSInstance(data.x[i], i);
            this.data.push(e);
        }

        this.init();
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

    default_weight(distance) {
        return 1;
    }

    gaussian_kernel_weight(distance) {
        let val = ( 1 / (this.bandwidth * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * Math.pow(distance / this.bandwidth, 2));
        return val;
    }

    updated(old_n, new_n) {
        let o = [];
        let n = [];
        for (let i = 0; i < old_n.length; i++) {
            o.push(old_n.id);
        }
        for (let i = 0; i < new_n.length; i++) {
            n.push(new_n.id);
        }
        o.sort();
        n.sort();

        if (o.length != n.length) {
            return true;
        }
        for (let i = 0; i < o.length; i++) {
            if (o[i] != n[i]) {
                return true;
            }
        }
        return false;
    }

    // Find neighbors within the bandwidth of this centroid
    find_neighbors(ctr) {
        let weight_func = this.gaussian_kernel_weight;

        let neighbors = [];
        let new_x = new Array(ctr.no_attr()).fill(0);
        let den = 0;

        for (let i = 0; i < this.data.length; i++) {
            let inst = this.data[i];
            let dist = this.dist_func(inst, ctr);
            // Instance is within bandwidth distance
            if (dist <= this.bandwidth) {
                neighbors.push(inst);

                let w = 1;
                if (this.weightf == 1) {
                    w = this.gaussian_kernel_weight(dist);
                }

                // Update new x
                for (let a = 0; a < ctr.no_attr(); a++) {
                    new_x[a] += inst.x[a] * w;
                }
                den += w;
            }
        }

        // Check if updated
        let upd = this.updated(neighbors, ctr.neighbors);

        // Set neighbors
        ctr.neighbors = neighbors;

        // Average center point values
        for (let a = 0; a < ctr.no_attr(); a++) {
            new_x[a] /= den;
        }

        // Set new center point
        ctr.x = new_x;

        return upd;
    }

    // Remove duplicates to find unique clusters
    unique() {
        let new_centroids = [];

        // Iterate over all centroids
        for (let i = 0; i < this.centroids.length; i++) {
            let c1 = this.centroids[i];

            // Iterate over the new centroid list without duplicates
            let found = false;
            for (let j = 0; j < new_centroids.length; j++) {
                let c2 = new_centroids[j];
                // Check if distance is below a min distance
                let d = this.dist_func(c1, c2);
                if (d <= 0.2) {
                    // Centroids are close - merge them
                    found = true;
                    c2.merge(c1);
                    break;
                }
            }
            // New centroid - add it
            if (!found) {
                c1.set_assigned();
                new_centroids.push(c1);
            }
        }

        // Set new centroids
        this.centroids = new_centroids;

        // Set label
        for (let i = 0; i < this.centroids.length; i++) {
            this.centroids[i].label = i + 1;
            this.centroids[i].move_to_center();
        }
    }
    
    // Executes one training iteration
    iterate() {
        let old_x = [];
        let new_x = [];
        let updated = false;
        // Calculate neighbors for each centroid
        for (let c = 0; c < this.centroids.length; c++) {
            let ctr = this.centroids[c];
            old_x.push(ctr.x); // Old centroid center
            let upd = this.find_neighbors(ctr);
            if (upd) {
                updated = upd;
            }
            new_x.push(ctr.x); // Updated centroid center
        }
        
        // Stop training if no new assignments were made
        // or if max iterations are reached
        if (!updated || this.current_iter >= this.max_iter) {
            this.training_done = true;
            // Remove duplicate centroids to create
            // nice clusters
            this.unique();
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
        let inst = new MSInstance(i);
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
class MSInstance {
    constructor(x, id) {
        this.x = x;
        this.id = id;
        this.assigned = false;
    }

    // Sets this instance as assigned to a cluster
    set_assigned() {
        this.assigned = true;
    }
}

/**
    Centroid class for K-Means.
*/
class MSCentroid {
    // Constructor
    constructor(x) {
        this.x = x;
        this.label = -1;
        this.neighbors = [];
    }

    // Number of attributes
    no_attr() {
        return this.x.length;
    }

    // Check assignments for this centroid. Already assigned
    // instances are removed.
    set_assigned() {
        let new_neighbors = [];
        for (let i = 0; i < this.neighbors.length; i++) {
            let inst = this.neighbors[i];
            if (!inst.assigned) {
                inst.set_assigned();
                new_neighbors.push(inst);
            }
        }
        this.neighbors = new_neighbors;
    }

    // Add all neighbors from another cluster to this cluster
    merge(c2) {
        for (let i = 0; i < c2.neighbors.length; i++) {
            let ninst = c2.neighbors[i]; 
            // Instance not assigned - add it
            if (!ninst.assigned) {
                ninst.set_assigned();
                this.neighbors.push(ninst);
            }
        }
    }

    // Move this centroid to the center of all assigned neighbors
    move_to_center() {
        let new_x = new Array(this.no_attr()).fill(0);

        for (let i = 0; i < this.neighbors.length; i++) {
            let inst = this.neighbors[i];
            // Update new x
            for (let a = 0; a < this.no_attr(); a++) {
                new_x[a] += inst.x[a];
            }
        }
        
        // Average center point values
        for (let a = 0; a < this.no_attr(); a++) {
            new_x[a] /= this.neighbors.length;
        }

        // Set new center point
        this.x = new_x;
    }
}
