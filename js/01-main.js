
/** ------------------------------------------------------

Main functions for the Web Clustering demo.

--------------------------------------------------------- */

/**
    Runs the visualization demo.
*/
async function demo() {
    running = true;
    
    // Wait function
    const delay = ms => new Promise(res => setTimeout(res, ms));
    
    // Disable button
    disable("demo", "running");
    await delay(25);
    
    clear();
    
    // Get clustering algorithm type and dataset name
    let type = get_selected_algorithm();
    let name = get_selected_dataset();

    // Get render options for the dataset
    let opt = get_render_options(name);
    let msize = get_marker_size(type);

    // Get data
    let data = orig_data.clone();

    // Build and train a clustering algorithm
    let cl = build_cluster(type);
    cl.train(data);
    
    if (!cl.iterable()) {
        let map = create_decision_boundaries(cl, opt);
        draw_map(map);
        draw_labels(data, opt);
        draw_markers(cl.get_markers(), opt, msize);
    }
    else {
        // Enable Stop button
        enable("stop");
        
        while (!cl.done() && running) {
            // Start time
            let start = new Date().getTime();

            // Iterate classifier
            cl.iterate();
            let map = create_decision_boundaries(cl, opt);
            draw_map(map);
            draw_labels(data, opt);
            draw_markers(cl.get_markers(), opt, msize);
            document.getElementById("citer").innerHTML = "Iteration: " + cl.current_iteration();
            
            // Time elapsed
            let end = new Date().getTime();
            let time = end - start;
            let rest = 150 - time;
            if (rest < 10) rest = 10;
            // Wait
            await delay(rest);
        }

        // Disable Stop button
        disable("stop");
    }
    document.getElementById("clusters").innerHTML = "Clusters: " + cl.no_clusters();
    
    // Enable button
    enable("demo");

    running = false;
}

/**
    Stops the iteration for iterable algorithms.
*/
function stop_demo() {
    running = false;
}

/**
    Optimize the k-value for K-Means clustering.
*/
async function optimize_k() {
    running = true;
    
    // Wait function
    const delay = ms => new Promise(res => setTimeout(res, ms));
    
    // Disable button
    disable("demo", "running");
    await delay(25);
    
    // Get dataset name
    let name = get_selected_dataset();
    // Get data
    let data = orig_data.clone();

    // Get params
    let dist = get_select_list("dist", 0);
    let init = get_select_list("init", 1);
    let seed = get_value("seed", 42, 0, 10000);

    // Run silhouette score optimization
    let opt = new KOptimizer(data, dist, init, seed);
    let k = opt.optimize();
    
    document.getElementById("k").value = k;

    // Enable button
    enable("demo");

    running = false;
}

/**
    Creates a map of the dedision boundaries for the current trained clustering algorithm.
*/
function create_decision_boundaries(classifier, opt) {
    let map = new Array(100 * 100).fill(0);
    
    // 100x100 map
    for (let x1 = 0; x1 < 100; x1++) {
        for (let x2 = 0; x2 < 100; x2++) {
            // x-values
            let v1 = x1 / 100.0 * opt[0] + opt[2];
            let v2 = x2 / 100.0 * opt[1] + opt[3];
            
            v1 += 0.005;
            v2 += 0.005;
            
            // Prediction
            let pred = classifier.predict([[v1, v2]]);
            // Set predicted label
            map[x1 + x2 * 100] = pred[0];
        }
    }
    return map;
}

/**
    Validates hyperparameter settings.
*/
function validate_setting(id, value, min_val, max_val) {
    let v = value;
    if (v < min_val) {
        v = min_val;
    }
    if (v > max_val) {
        v = max_val;
    }
    if (v != value) {
        document.getElementById(id).value = v;
    }
    return v;
}

/**
    Builds the specified clustering algorithm and trains on the specified dataset.
*/
function build_cluster(type) {
    if (type == "kmeans") {
        // Get options
        let k = get_value("k", 3, 1, 32);
        let dist = get_select_list("dist", 0);
        let init = get_select_list("init", 1);
        let seed = get_value("seed", 42, 0, 10000);

        let cl = new KMeans(k, dist, init, seed);
        return cl;
    }
    else if (type == "dbscan") {
        // Get options
        let eps = get_value("eps", 0.2, 0.01, 100, parseFloat);
        let min_pts = get_value("minpts", 5, 2, 100);
        let dist = get_select_list("dist", 0);
        
        let cl = new DBScan(eps, min_pts, dist);
        return cl;
    }
    else if (type == "meanshift") {
        // Get options
        let bw = get_value("bandwidth", 0.1, 0.05, 1, parseFloat);
        let dist = get_select_list("dist", 0);
        let w = get_select_list("weight", 0);
        
        let cl = new MeanShift(bw, dist, w);
        return cl;
    }
    else {
        throw("Unknown clustering algorithm: " + type);
    }
}

/** ------------------------------------------------------

Util functions for the VisualML demo.

--------------------------------------------------------- */

// Is set to true if a classifier is running
var running = false;

/**
    Shows the element with the specified id.
*/
function show(id) {
    let e = document.getElementById(id);
    if (e != null) {
        e.style.display = "block";
    }
}

/**
    Hides the element with the specified id.
*/
function hide(id) {
    let e = document.getElementById(id);
    if (e != null) {
        e.style.display = "none";
    }
}

/**
    Toggles visibility of the element with the specified id.
*/
function toggle(id) {
    let e = document.getElementById(id);
    if (e.style.display == "none") {
        e.style.display = "block";
    }
    else {
        e.style.display = "none";
    }
}

/**
    Toggles visibility of the element with the specified id and updates 
    expand arrow.
*/
function toggle_bt(id) {
    let e = document.getElementById(id);
    let bt = document.getElementById(id + "_bt");
    if (e.style.display == "none") {
        e.style.display = "block";
        bt.innerHTML = "&#9660;";
    }
    else {
        e.style.display = "none";
        bt.innerHTML = "&#9658;";
    }
}

/**
    Enables an element.
*/
function enable(id) {
    let e = document.getElementById(id);
    e.disabled = false;
    e.className = "enabled";
}

/**
    Disables an element.
*/
function disable(id, classid="disabled") {
    let e = document.getElementById(id);
    e.disabled = true;
    e.className = classid;
}

/**
    Returns the value of an input field as an integer array.
*/
function get_array(id, default_arr, conv_func=parseInt) {
    let e = document.getElementById(id);
    if (e == null) {
        return default_arr;
    }
    
    let str = e.value;
    
    let arr = str.split(",");
    for (let i in arr) {
        let val = arr[i];
        val = val.trim();
        arr[i] = conv_func(val);
        if (isNaN(arr[i])) {
            e.value = default_arr;
            return default_arr;
        }
    }
    return arr;
}

/**
    Returns the integer value from an input field.
*/
function get_value(id, default_val, min_val, max_val, conv_func=parseInt) {
    // Check if element is available
    let e = document.getElementById(id);
    if (e == null) {
        return default_val;
    }
    
    let str = e.value;
    str = str.trim();
    let val = conv_func(str);
    // Check if valid int
    if (isNaN(val)) {
        e.value = default_val;
        val = default_val;
    }

    // Range check
    if (val < min_val) {
        val = min_val;
        e.value = val;
    }
    if (val > max_val) {
        val = max_val;
        e.value = val;
    }

    return val;
}

/**
    Returns the selected value in a dropdown list.
*/
function get_select_list(id, default_val) {
    // Check if element is available
    let e = document.getElementById(id);
    if (e == null) {
        return default_val;
    }

    // Get selected value
    let val = e.options[e.selectedIndex].value;
    return val;
}


/**
    Returns the selected classifier.
*/
function get_selected_algorithm() {
    // Dataset radio buttons
    let name = ""
    let rlist = document.getElementsByName("sel-cl");
    for (let i = 0, length = rlist.length; i < length; i++) {
        if (rlist[i].checked)  {
            name = rlist[i].value;
            break;
        }
    }
    return name;
}

/**
    Returns the selected dataset.
*/
function get_selected_dataset() {
    // Dataset radio buttons
    let name = ""
    let rlist = document.getElementsByName("sel-ds");
    for (let i = 0, length = rlist.length; i < length; i++) {
        if (rlist[i].checked)  {
            name = rlist[i].value;
            break;
        }
    }
    return name;
}

/**
    Shows the settings for the selected classifier.
*/
function update_settings() {
    let type = get_selected_algorithm();
    let name = get_selected_dataset();
    
    let settings = get_settings(type, name);
    
    document.getElementById("opts").innerHTML = "";
    
    let html = "<table><tr>";
    if (type == "kmeans") {
        html += "<td class='param'>Clusters:</td><td><input class='value' name='k' id='k' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Distance function:</td><td><select id='dist' class='value'><option value='0'>Euclidean</option><option value='1'>Manhattan</option><option value='2'>Chebyshev</option></select></td>";
        html += "<td class='param'>Initialization:</td><td><select id='init' class='value'><option value='1'>K-Means++</option><option value='0'>Random</option></select></td>";
        html += "<td class='param'>Seed:</td><td><input class='value' name='seed' id='seed' value='" + settings[1] + "'></td>";
        html += "</tr><tr>";
        html += "<td colspan='2' class='param'><input type='button' style='background-color: #d4e3cf;' onclick='javascript:optimize_k()' value='Find no clusters'></button></td>";
        html += "<td colspan='6'>Find optimal number of clusters using silhouette score</td>";
    }
    else if (type == "dbscan") {
        html += "<td class='param'>Epsilon:</td><td><input class='value' name='eps' id='eps' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Min samples:</td><td><input class='value' name='minpts' id='minpts' value='" + settings[1] + "'></td>";
        html += "<td class='param'>Distance function:</td><td><select id='dist' class='value'><option value='0'>Euclidean</option><option value='1'>Manhattan</option><option value='2'>Chebyshev</option></select></td>";
    }
    else if (type == "meanshift") {
        html += "<td class='param'>Bandwidth:</td><td><input class='value' name='bandwidth' id='bandwidth' value='" + settings[0] + "'></td>";
        html += "<td class='param'>Distance function:</td><td><select id='dist' class='value'><option value='0'>Euclidean</option><option value='1'>Manhattan</option><option value='2'>Chebyshev</option></select></td>";
        html += "<td class='param'>Weight:</td><td><select id='weight' class='value'><option value='1'>Gaussian Kernel</option><option value='0'>None</option></select></td>";
    }
    html += "</tr></table>";
    
    document.getElementById("opts").innerHTML = html;
}

/**
    Shows the dataset labels.
*/
function show_data() {
    if (running) return;

    clear();
    
    data_name = get_selected_dataset();
    orig_data = get_dataset(data_name);
    let opt = get_render_options(data_name);
    
    draw_labels(orig_data, opt);
}

/**
    Shows the library version in a html "version" field.
*/
function show_version() {
    document.getElementById("version").innerHTML = VERSION;
}

