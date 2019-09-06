
/** ------------------------------------------------------

Implementation of DBSCAN clustering algorithm.
See: http://ros-developer.com/2017/12/09/density-based-spatial-clustering-dbscan-with-python-code/

--------------------------------------------------------- */

class DBScan {
    constructor(eps, min_pts, distf) {
        this.data = [];
        this.cores = [];
        this.clusters = [];
        this.eps = eps;
        this.min_pts = min_pts;
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
    }

    // Returns number of clusters
    no_clusters() {
        return this.clusters.length;
    }
    
    // Sets if this classifier has iterable training phase
    iterable() {
        return false;
    }

    // Returns the neighbors within eps distance from the specified instance
    get_neighbors(P) {
        let neighbors = [];

        for (let i = 0; i < this.data.length; i++) {
            let N = this.data[i];
            if (this.dist_func(P, N) <= this.eps) {
                neighbors.push(N);
            }
        }

        return neighbors;
    }

    // Runs the DBSCAN algorithm
    run_dbscan() {
        this.clusters = [];

        for (let i = 0; i < this.data.length; i++) {
            let P = this.data[i];
            if (P.status == DBInstance.UNVISITED) {
                P.status = DBInstance.VISITED;
                let N = this.get_neighbors(P);
                if (N.length < this.min_pts) {
                    P.status = DBInstance.NOISE;
                }
                else {
                    P.is_core = true;
                    let C = new DBCluster(this.clusters.length + 1);
                    this.clusters.push(C);
                    this.expand_cluster(P, N, C);
                }
            }
        }
    }

    // Expands a cluster with new instances
    expand_cluster(P, N, C) {
        C.add(P);
        for (let i = 0; i < N.length; i++) {
            let Pi = N[i];
            if (Pi.status == DBInstance.UNVISITED) {
                Pi.status = DBInstance.VISITED;
                let Ni = this.get_neighbors(Pi);
                if (Ni.length >= this.min_pts) {
                    Pi.is_core = true;
                    this.merge(N, Ni);
                }
            }
            if (!Pi.is_member) {
                C.add(Pi);
            }
        }
    }

    // Merges two arrays of instances
    merge(N, Ni) {
        for (let i = 0; i < Ni.length; i++) {
            N.push(Ni[i]);
        }
    }

    // Trains the clustering algorithm
    train(data) {
        data.shuffle();

         // Add data to internal array
        for (let i = 0; i < data.no_examples(); i++) {
            let e = new DBInstance(data.x[i]);
            this.data.push(e);
        }

        // Find core instances
        this.run_dbscan();
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
        let inst = new DBInstance(i);

        for (let j = 0; j < this.clusters.length; j++) {
            let C = this.clusters[j];
            for (let i = 0; i < C.points.length; i++) {
                let N = C.points[i];
                if (N.is_core) {
                    if (this.dist_func(inst, N) <= this.eps) {
                        return C.label;
                    }
                }
            }
        }

        return 0;
    }

    // Returns the markers to be rendered, nothing in this case
    get_markers() {
        let m = [];

        for (let i = 0; i < this.data.length; i++) {
            let inst = this.data[i];
            if (inst.is_core) {
                m.push(inst.x);
            }
        }
        
        return m;
    }
    
    // Calculates distance (Euclidean) between two instances
    dist_euclidean(inst1, inst2) {
        let sumSq = 0;
        for (let i = 0; i < inst1.no_attr(); i++) {
            sumSq += Math.pow(inst1.x[i] - inst2.x[i], 2);
        }
        sumSq = Math.sqrt(sumSq);
        return sumSq;
    }

    // Calculates Manhattan distance between two instances
    dist_manhattan(inst1, inst2) {
        let sum = 0;
        for (let i = 0; i < inst1.no_attr(); i++) {
            sum += Math.abs(inst1.x[i] - inst2.x[i]);
        }
        return sum;
    }

    // Calculates Chebyshev distance between two instances
    dist_chebyshev(inst1, inst2) {
        let best_v = 0;
        for (let i = 0; i < inst1.no_attr(); i++) {
            let v = Math.abs(inst1.x[i] - inst2.x[i]);
            if (v > best_v) {
                best_v = v;
            }
        }
        return best_v;
    }
}

/**
    Holds a DBSCAN cluster of instances.
*/
class DBCluster {
    constructor(label) {
        this.points = [];
        this.label = label;
    }

    add(P) {
        this.points.push(P);
        P.is_member = true;
    }
}

/**
    Holds a dataset instance.
*/
class DBInstance {
    constructor(x) {
        this.x = x;
        this.status = DBInstance.UNVISITED;
        this.is_member = false;
        this.is_core = false;
    }

    no_attr() {
        return this.x.length;
    }
}

DBInstance.UNVISITED = 0;
DBInstance.VISITED = 1;
DBInstance.NOISE = 2;
