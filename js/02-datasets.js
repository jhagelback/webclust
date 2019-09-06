
// Randomizer seed
var global_seed = 42;
// Original data
var orig_data = null;
// Dataset name
var data_name;

/**
	Custom random function.

	Built-in random has no seed feature. Instead a simple pseudo-random with seed is used.  
	See explanation here:
    https://stackoverflow.com/questions/521295/seeding-the-random-number-generator-in-javascript
*/
function rnd() {
    let x = Math.sin(global_seed++) * 10000;
    return x - Math.floor(x);
}

/**
    Random normally distributed using the Box-Muller transform.

    See explanation here:
    https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
*/
function rnd_bm() {
    let u = 0, v = 0;
    while(u === 0) u = rnd(); //Converting [0,1) to (0,1)
    while(v === 0) v = rnd();
    let num = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
    num = num / 10.0 + 0.5; // Translate to 0 -> 1
    if (num > 1 || num < 0) return rnd_bm(); // resample between 0 and 1
    return num;
}

/**
    Dictionary get function.
*/
function get(dict, key) {
    key += "";
    for (let  i = 0; i < dict.length; i++) {
        if (dict[i].key == key) {
            return dict[i].value;
        }
    }
    return null;
}

/** ------------------------------------------------------

Datasets used in the VisualML demo.

--------------------------------------------------------- */

/**
    Spiral dataset.
*/
function get_spiral() {
    let spiral_x = [
	    [ 0.000, 0.000], [ 0.001, 0.010], [ 0.006, 0.019], [ 0.016, 0.026], [ 0.021, 0.035], [ 0.000, 0.051], [ 0.025, 0.055], [ 0.018, 0.068],
		[ 0.024, 0.077], [ 0.039, 0.082], [ 0.042, 0.092], [ 0.075, 0.082], [ 0.072, 0.097], [ 0.069, 0.112], [ 0.086, 0.112], [ 0.094, 0.118],
		[ 0.131, 0.095], [ 0.103, 0.137], [ 0.129, 0.128], [ 0.108, 0.159], [ 0.059, 0.193], [ 0.176, 0.118], [ 0.194, 0.108], [ 0.164, 0.165],
		[ 0.240, 0.036], [ 0.166, 0.190], [ 0.229, 0.128], [ 0.237, 0.135], [ 0.280, 0.037], [ 0.291, 0.031], [ 0.287, 0.098], [ 0.304, 0.075],
		[ 0.290, 0.142], [ 0.269, 0.197], [ 0.331, 0.090], [ 0.351, 0.044], [ 0.361,-0.047], [ 0.369,-0.061], [ 0.381, 0.043], [ 0.393, 0.022],
		[ 0.399, 0.066], [ 0.406, 0.082], [ 0.414, 0.091], [ 0.369,-0.230], [ 0.442,-0.047], [ 0.449,-0.072], [ 0.464,-0.017], [ 0.420,-0.221],
		[ 0.484,-0.022], [ 0.462,-0.177], [ 0.487,-0.135], [ 0.434,-0.277], [ 0.478,-0.218], [ 0.506,-0.176], [ 0.449,-0.310], [ 0.411,-0.373],
		[ 0.431,-0.367], [ 0.404,-0.410], [ 0.468,-0.353], [ 0.440,-0.402], [ 0.456,-0.399], [ 0.419,-0.451], [ 0.449,-0.437], [ 0.514,-0.375],
		[ 0.321,-0.561], [ 0.368,-0.543], [ 0.479,-0.464], [ 0.227,-0.638], [ 0.374,-0.576], [ 0.235,-0.656], [ 0.118,-0.697], [ 0.175,-0.695],
		[ 0.003,-0.727], [ 0.313,-0.667], [ 0.053,-0.746], [ 0.186,-0.734], [ 0.186,-0.745], [ 0.113,-0.769], [ 0.041,-0.787], [-0.049,-0.796],
		[ 0.115,-0.800], [-0.251,-0.779], [-0.217,-0.799], [ 0.080,-0.835], [-0.443,-0.723], [-0.534,-0.672], [-0.468,-0.732], [-0.291,-0.829],
		[-0.176,-0.871], [-0.555,-0.707], [-0.366,-0.832], [-0.646,-0.654], [-0.538,-0.758], [-0.681,-0.647], [-0.632,-0.709], [-0.713,-0.642],
		[-0.653,-0.717], [-0.888,-0.414], [-0.739,-0.658], [-0.807,-0.591], [-0.000,-0.000], [-0.006,-0.008], [-0.013,-0.016], [-0.028,-0.012],
		[-0.029,-0.029], [-0.050,-0.006], [-0.052,-0.032], [-0.059,-0.039], [-0.081,-0.000], [-0.091,-0.005], [-0.101, 0.007], [-0.111,-0.010],
		[-0.112,-0.047], [-0.129, 0.025], [-0.139,-0.028], [-0.151, 0.008], [-0.160, 0.020], [-0.171,-0.010], [-0.180, 0.025], [-0.186, 0.046],
		[-0.199, 0.034], [-0.211,-0.018], [-0.216, 0.052], [-0.206, 0.108], [-0.241, 0.029], [-0.244, 0.067], [-0.254, 0.065], [-0.200, 0.186],
		[-0.241, 0.149], [-0.251, 0.151], [-0.285, 0.103], [-0.250, 0.189], [-0.292, 0.139], [-0.270, 0.196], [-0.296, 0.175], [-0.237, 0.263],
		[-0.238, 0.275], [-0.276, 0.252], [-0.238, 0.301], [-0.315, 0.237], [-0.332, 0.230], [-0.212, 0.355], [-0.223, 0.361], [-0.177, 0.397],
		[-0.013, 0.444], [-0.124, 0.437], [-0.265, 0.382], [-0.076, 0.469], [-0.277, 0.398], [-0.191, 0.457], [-0.138, 0.486], [ 0.062, 0.511],
		[-0.171, 0.497], [-0.162, 0.510], [-0.066, 0.541], [-0.107, 0.545], [ 0.115, 0.554], [-0.112, 0.565], [-0.099, 0.577], [ 0.008, 0.596],
		[ 0.025, 0.606], [ 0.331, 0.520], [ 0.251, 0.574], [ 0.176, 0.612], [ 0.037, 0.645], [ 0.322, 0.572], [ 0.122, 0.655], [ 0.078, 0.672],
		[ 0.443, 0.525], [ 0.375, 0.587], [ 0.471, 0.527], [ 0.434, 0.571], [ 0.520, 0.508], [ 0.377, 0.634], [ 0.358, 0.656], [ 0.586, 0.481],
		[ 0.450, 0.622], [ 0.495, 0.600], [ 0.553, 0.562], [ 0.631, 0.488], [ 0.622, 0.516], [ 0.536, 0.618], [ 0.649, 0.514], [ 0.491, 0.680],
		[ 0.801, 0.279], [ 0.634, 0.579], [ 0.717, 0.491], [ 0.832, 0.283], [ 0.802, 0.384], [ 0.898,-0.045], [ 0.808, 0.417], [ 0.912, 0.114],
		[ 0.920, 0.134], [ 0.889, 0.304], [ 0.948,-0.046], [ 0.958, 0.048], [ 0.954,-0.173], [ 0.954,-0.223], [ 0.850,-0.507], [ 0.916,-0.402],
		[ 0.000,-0.000], [ 0.010,-0.001], [ 0.018,-0.009], [ 0.028,-0.012], [ 0.037,-0.017], [ 0.050,-0.001], [ 0.056,-0.023], [ 0.068,-0.020],
		[ 0.070,-0.041], [ 0.080,-0.043], [ 0.075,-0.068], [ 0.088,-0.068], [ 0.085,-0.086], [ 0.110,-0.072], [ 0.123,-0.071], [ 0.119,-0.094],
		[ 0.114,-0.115], [ 0.106,-0.135], [ 0.044,-0.176], [ 0.119,-0.151], [ 0.146,-0.140], [ 0.128,-0.169], [ 0.131,-0.180], [ 0.090,-0.214],
		[ 0.168,-0.175], [ 0.099,-0.232], [ 0.088,-0.247], [ 0.077,-0.262], [ 0.113,-0.259], [ 0.087,-0.280], [ 0.145,-0.266], [ 0.084,-0.302],
		[ 0.077,-0.314], [ 0.003,-0.333], [ 0.096,-0.330], [-0.051,-0.350], [-0.117,-0.344], [ 0.126,-0.352], [-0.075,-0.377], [-0.111,-0.378],
		[-0.026,-0.403], [-0.063,-0.409], [-0.103,-0.412], [-0.109,-0.421], [-0.128,-0.426], [-0.026,-0.454], [-0.286,-0.366], [-0.302,-0.366],
		[-0.167,-0.455], [-0.128,-0.478], [-0.325,-0.386], [-0.256,-0.447], [-0.340,-0.400], [-0.325,-0.425], [-0.426,-0.341], [-0.447,-0.329],
		[-0.361,-0.435], [-0.326,-0.475], [-0.332,-0.483], [-0.526,-0.280], [-0.418,-0.439], [-0.497,-0.364], [-0.513,-0.359], [-0.570,-0.283],
		[-0.452,-0.462], [-0.621,-0.214], [-0.650,-0.148], [-0.652,-0.181], [-0.654,-0.210], [-0.685,-0.130], [-0.704,-0.062], [-0.557,-0.452],
		[-0.694, 0.218], [-0.737, 0.024], [-0.741,-0.101], [-0.757,-0.033], [-0.756, 0.133], [-0.774, 0.072], [-0.763,-0.195], [-0.654, 0.457],
		[-0.790, 0.172], [-0.725, 0.379], [-0.815, 0.148], [-0.660, 0.517], [-0.762, 0.373], [-0.727, 0.456], [-0.837, 0.232], [-0.636, 0.607],
		[-0.681, 0.571], [-0.586, 0.682], [-0.791, 0.448], [-0.766, 0.508], [-0.352, 0.860], [-0.801, 0.491], [-0.678, 0.665], [-0.461, 0.842],
		[-0.601, 0.761], [-0.496, 0.845], [-0.628, 0.766], [-0.473, 0.881]
	];
    
    let data = new Dataset(spiral_x);
    return data;
}

/**
    Flame dataset.
*/
function get_flame() {
    let flame_x = [
		[0.123,0.273], [0.090,0.312], [0.093,0.425], [0.057,0.432], [0.033,0.455], [0.043,0.488], [0.073,0.465], [0.090,0.445],
		[0.130,0.440], [0.160,0.452], [0.120,0.467], [0.167,0.472], [0.197,0.487], [0.127,0.492], [0.090,0.485], [0.090,0.503],
		[0.083,0.522], [0.117,0.532], [0.133,0.513], [0.167,0.500], [0.113,0.565], [0.160,0.532], [0.203,0.518], [0.247,0.518],
		[0.230,0.537], [0.197,0.550], [0.160,0.553], [0.160,0.592], [0.190,0.575], [0.217,0.565], [0.263,0.547], [0.180,0.607],
		[0.230,0.598], [0.253,0.582], [0.267,0.563], [0.297,0.537], [0.310,0.562], [0.323,0.585], [0.287,0.598], [0.223,0.623],
		[0.247,0.657], [0.293,0.635], [0.283,0.620], [0.320,0.612], [0.350,0.592], [0.383,0.582], [0.353,0.558], [0.403,0.548],
		[0.433,0.570], [0.403,0.593], [0.373,0.607], [0.363,0.628], [0.337,0.648], [0.303,0.665], [0.330,0.685], [0.390,0.707],
		[0.373,0.690], [0.377,0.667], [0.397,0.640], [0.417,0.653], [0.407,0.618], [0.440,0.612], [0.443,0.590], [0.487,0.588],
		[0.523,0.590], [0.477,0.607], [0.507,0.610], [0.447,0.625], [0.487,0.625], [0.447,0.640], [0.487,0.645], [0.450,0.657],
		[0.493,0.660], [0.437,0.675], [0.490,0.673], [0.453,0.702], [0.497,0.697], [0.457,0.718], [0.507,0.713], [0.570,0.712],
		[0.547,0.683], [0.527,0.663], [0.537,0.650], [0.520,0.633], [0.533,0.618], [0.560,0.597], [0.577,0.608], [0.593,0.630],
		[0.560,0.630], [0.577,0.645], [0.563,0.665], [0.590,0.688], [0.640,0.690], [0.610,0.667], [0.680,0.667], [0.633,0.645],
		[0.717,0.647], [0.697,0.627], [0.657,0.630], [0.627,0.613], [0.677,0.610], [0.657,0.595], [0.603,0.592], [0.620,0.577],
		[0.610,0.562], [0.567,0.573], [0.777,0.618], [0.740,0.612], [0.693,0.592], [0.667,0.568], [0.797,0.592], [0.750,0.587],
		[0.707,0.570], [0.743,0.567], [0.793,0.572], [0.840,0.570], [0.787,0.552], [0.737,0.552], [0.687,0.553], [0.660,0.542],
		[0.697,0.533], [0.870,0.537], [0.833,0.542], [0.793,0.532], [0.747,0.525], [0.723,0.505], [0.760,0.492], [0.780,0.513],
		[0.820,0.518], [0.863,0.515], [0.837,0.502], [0.803,0.492], [0.783,0.463], [0.817,0.472], [0.853,0.483], [0.903,0.500],
		[0.907,0.480], [0.863,0.467], [0.833,0.458], [0.813,0.438], [0.847,0.422], [0.867,0.443], [0.903,0.460], [0.937,0.458],
		[0.947,0.432], [0.940,0.413], [0.900,0.440], [0.890,0.417], [0.887,0.400], [0.487,0.562], [0.530,0.555], [0.513,0.532],
		[0.450,0.537], [0.350,0.522], [0.410,0.510], [0.467,0.510], [0.507,0.493], [0.570,0.513], [0.623,0.517], [0.553,0.485],
		[0.527,0.480], [0.477,0.475], [0.447,0.490], [0.347,0.497], [0.413,0.468], [0.450,0.453], [0.410,0.450], [0.377,0.460],
		[0.310,0.448], [0.273,0.418], [0.357,0.440], [0.493,0.447], [0.517,0.463], [0.567,0.457], [0.620,0.467], [0.647,0.435],
		[0.587,0.435], [0.537,0.437], [0.507,0.428], [0.457,0.433], [0.413,0.425], [0.380,0.420], [0.340,0.415], [0.303,0.395],
		[0.367,0.400], [0.407,0.398], [0.433,0.413], [0.450,0.402], [0.487,0.408], [0.553,0.420], [0.593,0.410], [0.637,0.412],
		[0.690,0.397], [0.530,0.398], [0.263,0.387], [0.250,0.358], [0.260,0.335], [0.303,0.312], [0.350,0.308], [0.433,0.280],
		[0.497,0.280], [0.557,0.288], [0.617,0.293], [0.663,0.317], [0.703,0.347], [0.660,0.368], [0.613,0.383], [0.570,0.393],
		[0.587,0.373], [0.613,0.355], [0.637,0.332], [0.603,0.313], [0.587,0.340], [0.543,0.322], [0.537,0.340], [0.557,0.360],
		[0.527,0.357], [0.537,0.377], [0.487,0.387], [0.503,0.372], [0.457,0.385], [0.417,0.378], [0.370,0.383], [0.310,0.363],
		[0.333,0.348], [0.370,0.330], [0.370,0.358], [0.413,0.360], [0.453,0.365], [0.493,0.358], [0.443,0.352], [0.410,0.340],
		[0.433,0.330], [0.440,0.313], [0.513,0.312], [0.500,0.327], [0.500,0.345], [0.470,0.338], [0.460,0.295], [0.410,0.303]
    ];
    
    let data = new Dataset(flame_x);
    return data;
}

/**
    Moons dataset.
*/
function get_moons() {
    let moons_x = [
		[0.019,0.412], [0.017,0.453], [0.073,0.457], [0.117,0.484], [0.109,0.452], [0.119,0.448], [0.113,0.402], [0.102,0.394],
		[0.090,0.383], [0.076,0.362], [0.064,0.330], [0.069,0.314], [0.087,0.314], [0.098,0.354], [0.160,0.478], [0.170,0.433],
		[0.158,0.386], [0.157,0.358], [0.130,0.343], [0.122,0.316], [0.146,0.316], [0.134,0.304], [0.116,0.280], [0.101,0.269],
		[0.113,0.258], [0.180,0.214], [0.226,0.184], [0.217,0.233], [0.204,0.331], [0.249,0.293], [0.280,0.287], [0.294,0.278],
		[0.259,0.203], [0.277,0.188], [0.296,0.181], [0.304,0.183], [0.314,0.202], [0.312,0.210], [0.337,0.262], [0.338,0.250],
		[0.271,0.336], [0.270,0.323], [0.283,0.310], [0.292,0.314], [0.306,0.311], [0.310,0.296], [0.320,0.297], [0.316,0.308],
		[0.313,0.317], [0.312,0.324], [0.382,0.249], [0.393,0.248], [0.390,0.240], [0.378,0.203], [0.368,0.198], [0.426,0.237],
		[0.418,0.251], [0.476,0.226], [0.351,0.326], [0.369,0.330], [0.388,0.339], [0.400,0.334], [0.406,0.351], [0.400,0.304],
		[0.413,0.306], [0.427,0.312], [0.432,0.309], [0.447,0.320], [0.447,0.336], [0.442,0.348], [0.432,0.377], [0.428,0.384],
		[0.473,0.304], [0.509,0.274], [0.514,0.264], [0.539,0.292], [0.490,0.350], [0.466,0.394], [0.481,0.417], [0.479,0.429],
		[0.480,0.438], [0.478,0.456], [0.498,0.433], [0.494,0.398], [0.514,0.377], [0.522,0.360], [0.528,0.351], [0.559,0.360],
		[0.567,0.368], [0.511,0.400], [0.532,0.406], [0.576,0.410], [0.614,0.452], [0.513,0.476], [0.522,0.462], [0.534,0.469],
		[0.544,0.473], [0.314,0.414], [0.318,0.427], [0.318,0.450], [0.328,0.464], [0.341,0.456], [0.354,0.434], [0.367,0.421],
		[0.386,0.421], [0.381,0.438], [0.370,0.442], [0.367,0.463], [0.361,0.468], [0.356,0.483], [0.353,0.507], [0.337,0.532],
		[0.338,0.540], [0.378,0.452], [0.376,0.459], [0.386,0.457], [0.381,0.464], [0.384,0.469], [0.393,0.467], [0.378,0.476],
		[0.374,0.482], [0.369,0.488], [0.380,0.489], [0.388,0.486], [0.396,0.484], [0.391,0.492], [0.382,0.500], [0.383,0.508],
		[0.380,0.517], [0.377,0.526], [0.367,0.529], [0.361,0.522], [0.357,0.536], [0.370,0.558], [0.371,0.547], [0.377,0.550],
		[0.384,0.551], [0.401,0.536], [0.413,0.522], [0.420,0.532], [0.416,0.550], [0.399,0.558], [0.409,0.577], [0.388,0.569],
		[0.391,0.574], [0.393,0.581], [0.384,0.584], [0.377,0.584], [0.372,0.586], [0.440,0.579], [0.424,0.588], [0.389,0.616],
		[0.390,0.620], [0.397,0.632], [0.404,0.614], [0.429,0.598], [0.431,0.603], [0.423,0.603], [0.420,0.611], [0.413,0.626],
		[0.416,0.630], [0.430,0.618], [0.443,0.616], [0.444,0.602], [0.451,0.602], [0.457,0.604], [0.408,0.646], [0.414,0.647],
		[0.429,0.644], [0.424,0.648], [0.426,0.652], [0.471,0.604], [0.476,0.604], [0.469,0.622], [0.453,0.644], [0.456,0.659],
		[0.447,0.666], [0.454,0.686], [0.466,0.677], [0.466,0.662], [0.464,0.653], [0.468,0.644], [0.486,0.611], [0.487,0.618],
		[0.496,0.629], [0.486,0.652], [0.473,0.688], [0.502,0.651], [0.500,0.663], [0.526,0.640], [0.536,0.644], [0.488,0.693],
		[0.492,0.688], [0.499,0.682], [0.499,0.691], [0.504,0.678], [0.511,0.676], [0.516,0.682], [0.521,0.668], [0.528,0.668],
		[0.543,0.663], [0.547,0.657], [0.560,0.654], [0.579,0.658], [0.562,0.672], [0.541,0.681], [0.518,0.691], [0.510,0.694],
		[0.498,0.699], [0.507,0.709], [0.509,0.711], [0.517,0.714], [0.521,0.720], [0.523,0.707], [0.529,0.719], [0.529,0.694],
		[0.538,0.711], [0.546,0.711], [0.549,0.714], [0.549,0.704], [0.553,0.694], [0.587,0.673], [0.603,0.668], [0.607,0.679],
		[0.611,0.679], [0.612,0.687], [0.597,0.690], [0.591,0.691], [0.597,0.702], [0.582,0.702], [0.578,0.706], [0.559,0.709],
		[0.569,0.713], [0.574,0.720], [0.554,0.726], [0.558,0.728], [0.566,0.730], [0.597,0.734], [0.603,0.730], [0.604,0.733],
		[0.621,0.728], [0.621,0.722], [0.640,0.710], [0.640,0.696], [0.639,0.679], [0.636,0.672], [0.650,0.660], [0.667,0.654],
		[0.680,0.724], [0.668,0.723], [0.661,0.723], [0.649,0.711], [0.654,0.710], [0.646,0.699], [0.653,0.692], [0.656,0.696],
		[0.664,0.701], [0.683,0.701], [0.676,0.710], [0.684,0.712], [0.690,0.712], [0.687,0.684], [0.681,0.670], [0.682,0.663],
		[0.700,0.661], [0.703,0.654], [0.711,0.644], [0.722,0.623], [0.741,0.634], [0.724,0.646], [0.726,0.653], [0.723,0.659],
		[0.719,0.664], [0.723,0.671], [0.716,0.688], [0.719,0.706], [0.731,0.708], [0.727,0.698], [0.728,0.692], [0.758,0.698],
		[0.758,0.689], [0.747,0.683], [0.741,0.674], [0.750,0.668], [0.742,0.662], [0.766,0.671], [0.770,0.674], [0.770,0.661],
		[0.783,0.661], [0.763,0.649], [0.758,0.641], [0.766,0.638], [0.771,0.640], [0.774,0.644], [0.763,0.628], [0.768,0.626],
		[0.779,0.622], [0.789,0.621], [0.796,0.642], [0.813,0.651], [0.817,0.639], [0.811,0.636], [0.799,0.624], [0.802,0.620],
		[0.803,0.613], [0.836,0.637], [0.842,0.630], [0.648,0.702], [0.776,0.600], [0.784,0.591], [0.798,0.592], [0.800,0.586],
		[0.794,0.578], [0.816,0.597], [0.813,0.582], [0.820,0.583], [0.828,0.574], [0.809,0.574], [0.807,0.562], [0.817,0.559],
		[0.848,0.584], [0.853,0.590], [0.852,0.567], [0.838,0.560], [0.832,0.552], [0.830,0.547], [0.822,0.539], [0.818,0.529],
		[0.826,0.521], [0.828,0.530], [0.837,0.534], [0.843,0.537], [0.858,0.539], [0.856,0.529], [0.844,0.512], [0.829,0.511],
		[0.833,0.502], [0.841,0.478], [0.851,0.476], [0.846,0.479], [0.852,0.481], [0.856,0.483], [0.873,0.484], [0.867,0.507],
		[0.866,0.513], [0.871,0.526], [0.878,0.538], [0.879,0.527], [0.883,0.517], [0.893,0.516], [0.898,0.532], [0.899,0.522],
		[0.901,0.508], [0.899,0.478], [0.893,0.471], [0.903,0.469], [0.902,0.461], [0.918,0.460], [0.910,0.451], [0.917,0.427],
		[0.910,0.421], [0.904,0.434], [0.899,0.438], [0.887,0.440], [0.881,0.440], [0.872,0.456], [0.863,0.456], [0.851,0.433],
		[0.861,0.426], [0.867,0.431], [0.850,0.414], [0.878,0.423], [0.887,0.421]
    ];
    
    let data = new Dataset(moons_x);
    return data;
}

/**
    Gaussian dataset.
*/
function get_gaussian() {
	let gaussian_x = [
		[0.351,0.334], [0.282,0.315], [0.245,0.379], [0.373,0.209], [0.221,0.304], [0.384,0.295], [0.238,0.253], [0.246,0.111],
		[0.103,0.187], [0.250,0.363], [0.234,0.119], [0.349,0.257], [0.352,0.175], [0.284,0.280], [0.282,0.305], [0.437,0.209],
		[0.286,0.188], [0.216,0.258], [0.308,0.248], [0.331,0.278], [0.324,0.240], [0.111,0.214], [0.222,0.296], [0.316,0.238],
		[0.321,0.398], [0.276,0.332], [0.107,0.369], [0.386,0.350], [0.295,0.313], [0.402,0.265], [0.322,0.293], [0.313,0.329],
		[0.438,0.327], [0.305,0.104], [0.332,0.245], [0.252,0.211], [0.208,0.474], [0.295,0.330], [0.075,0.268], [0.088,0.389],
		[0.290,0.358], [0.295,0.341], [0.296,0.371], [0.259,0.155], [0.370,0.158], [0.268,0.155], [0.220,0.097], [0.417,0.334],
		[0.221,0.203], [0.233,0.124], [0.284,0.168], [0.154,0.316], [0.261,0.313], [0.309,0.270], [0.295,0.196], [0.354,0.334],
		[0.215,0.344], [0.331,0.057], [0.327,0.232], [0.272,0.313], [0.161,0.239], [0.411,0.407], [0.271,0.187], [0.225,0.246],
		[0.232,0.344], [0.281,0.262], [0.279,0.302], [0.248,0.167], [0.266,0.338], [0.222,0.192], [0.404,0.318], [0.410,0.274],
		[0.155,0.292], [0.263,0.293], [0.334,0.295], [0.352,0.307], [0.165,0.386], [0.196,0.316], [0.238,0.147], [0.225,0.291],
		[0.235,0.307], [0.255,0.339], [0.244,0.170], [0.202,0.188], [0.217,0.273], [0.306,0.383], [0.203,0.356], [0.285,0.337],
		[0.305,0.324], [0.220,0.280], [0.318,0.116], [0.257,0.389], [0.178,0.337], [0.232,0.194], [0.196,0.376], [0.234,0.400],
		[0.194,0.211], [0.195,0.097], [0.310,0.194], [0.129,0.170], [0.338,0.266], [0.228,0.265], [0.169,0.267], [0.289,0.289],
		[0.291,0.261], [0.365,0.341], [0.324,0.241], [0.288,0.179], [0.193,0.156], [0.419,0.371], [0.361,0.261], [0.305,0.322],
		[0.223,0.260], [0.355,0.387], [0.245,0.319], [0.291,0.336], [0.318,0.317], [0.336,0.383], [0.289,0.178], [0.190,0.405],
		[0.318,0.234], [0.320,0.268], [0.097,0.154], [0.344,0.233], [0.296,0.329], [0.708,0.766], [0.495,0.616], [0.662,0.731],
		[0.626,0.587], [0.718,0.621], [0.678,0.698], [0.529,0.707], [0.573,0.707], [0.759,0.639], [0.519,0.548], [0.667,0.550],
		[0.640,0.611], [0.571,0.555], [0.661,0.621], [0.581,0.782], [0.782,0.571], [0.690,0.533], [0.656,0.657], [0.681,0.685],
		[0.732,0.627], [0.717,0.621], [0.528,0.643], [0.617,0.682], [0.720,0.762], [0.682,0.614], [0.635,0.715], [0.641,0.594],
		[0.779,0.754], [0.693,0.556], [0.733,0.526], [0.692,0.634], [0.713,0.598], [0.633,0.768], [0.696,0.720], [0.591,0.752],
		[0.616,0.473], [0.791,0.643], [0.675,0.683], [0.596,0.779], [0.745,0.537], [0.659,0.552], [0.551,0.635], [0.587,0.738],
		[0.700,0.725], [0.713,0.678], [0.659,0.747], [0.605,0.523], [0.625,0.820], [0.572,0.585], [0.554,0.709], [0.643,0.671],
		[0.640,0.715], [0.724,0.727], [0.688,0.741], [0.572,0.642], [0.582,0.588], [0.669,0.701], [0.740,0.685], [0.682,0.705],
		[0.636,0.808], [0.723,0.794], [0.591,0.504], [0.744,0.696], [0.682,0.751], [0.612,0.678], [0.682,0.575], [0.655,0.657],
		[0.528,0.698], [0.689,0.680], [0.598,0.572], [0.492,0.747], [0.668,0.568], [0.614,0.628], [0.553,0.639], [0.755,0.777],
		[0.609,0.658], [0.663,0.599], [0.742,0.684], [0.662,0.744], [0.573,0.696], [0.762,0.591], [0.676,0.683], [0.505,0.735],
		[0.697,0.693], [0.594,0.689], [0.552,0.751], [0.650,0.799], [0.652,0.606], [0.615,0.738], [0.812,0.731], [0.682,0.513],
		[0.725,0.608], [0.602,0.627], [0.757,0.681], [0.759,0.681], [0.650,0.726], [0.613,0.599], [0.654,0.544], [0.772,0.671],
		[0.582,0.763], [0.657,0.431], [0.670,0.653], [0.617,0.523], [0.491,0.701], [0.709,0.685], [0.614,0.620], [0.709,0.659],
		[0.730,0.643], [0.699,0.412], [0.592,0.813], [0.588,0.589], [0.626,0.628], [0.786,0.731], [0.540,0.611], [0.611,0.643],
		[0.628,0.647], [0.584,0.429], [0.713,0.606], [0.707,0.447], [0.719,0.587], [0.675,0.606], [0.857,0.683], [0.719,0.805],
		[0.656,0.623], [0.622,0.685], [0.080,0.590], [0.163,0.596], [0.202,0.620], [0.182,0.794], [0.363,0.430], [0.303,0.739],
		[0.246,0.727], [0.311,0.741], [0.237,0.552], [0.407,0.621], [0.179,0.724], [0.180,0.568], [0.224,0.574], [0.063,0.689],
		[0.265,0.649], [0.312,0.761], [0.284,0.881], [0.387,0.677], [0.303,0.638], [0.221,0.675], [0.216,0.623], [0.281,0.703],
		[0.236,0.630], [0.380,0.620], [0.264,0.496], [0.358,0.657], [0.100,0.584], [0.282,0.654], [0.125,0.665], [0.270,0.629],
		[0.243,0.770], [0.306,0.722], [0.310,0.768], [0.208,0.727], [0.343,0.448], [0.287,0.608], [0.271,0.575], [0.186,0.637],
		[0.464,0.625], [0.222,0.736], [0.277,0.590], [0.424,0.788], [0.233,0.537], [0.413,0.550], [0.285,0.675], [0.231,0.625],
		[0.332,0.519], [0.234,0.756], [0.445,0.698], [0.314,0.680], [0.231,0.714], [0.110,0.382], [0.374,0.548], [0.328,0.609],
		[0.359,0.536], [0.113,0.648], [0.338,0.718], [0.298,0.504], [0.272,0.467], [0.172,0.740], [0.188,0.656], [0.413,0.817],
		[0.169,0.640], [0.218,0.642], [0.118,0.663], [0.246,0.716], [0.318,0.518], [0.418,0.579], [0.298,0.603], [0.408,0.681],
		[0.452,0.574], [0.210,0.709], [0.309,0.625], [0.276,0.741], [0.184,0.618], [0.334,0.776], [0.307,0.683], [0.416,0.647],
		[0.228,0.601], [0.196,0.728], [0.234,0.586], [0.357,0.649], [0.357,0.733], [0.237,0.654], [0.139,0.806], [0.123,0.578],
		[0.287,0.608], [0.205,0.525], [0.272,0.623], [0.493,0.616], [0.001,0.746], [0.151,0.816], [0.283,0.641], [0.080,0.679],
		[0.333,0.566], [0.302,0.568], [0.232,0.510], [0.289,0.465], [0.068,0.392], [0.294,0.753], [0.311,0.604], [0.171,0.450],
		[0.246,0.522], [0.272,0.409], [0.147,0.680], [0.240,0.709], [0.291,0.744], [0.367,0.849], [0.097,0.707], [0.407,0.626],
		[0.307,0.614], [0.337,0.958], [0.411,0.439], [0.395,0.655], [0.182,0.776], [0.315,0.712], [0.035,0.654], [0.139,0.660],
		[0.148,0.639], [0.355,0.472], [0.295,0.676], [0.224,0.684], [0.057,0.381], [0.150,0.567], [0.266,0.616], [0.591,0.243],
		[0.714,0.310], [0.735,0.106], [0.562,0.270], [0.614,0.385], [0.574,0.236], [0.619,0.330], [0.640,0.360], [0.628,0.287],
		[0.605,0.271], [0.690,0.206], [0.528,0.306], [0.583,0.163], [0.680,0.264], [0.467,0.250], [0.596,0.335], [0.609,0.190],
		[0.672,0.371], [0.669,0.277], [0.779,0.256], [0.669,0.398], [0.563,0.331], [0.547,0.265], [0.564,0.213], [0.627,0.302],
		[0.602,0.251], [0.512,0.190], [0.410,0.188], [0.619,0.294], [0.574,0.274], [0.614,0.305], [0.546,0.313], [0.639,0.189],
		[0.730,0.320], [0.633,0.300], [0.740,0.353], [0.972,0.274], [0.712,0.398], [0.604,0.250], [0.792,0.166], [0.627,0.276],
		[0.818,0.278], [0.624,0.297], [0.483,0.335], [0.537,0.253], [0.602,0.226], [0.763,0.251], [0.549,0.334], [0.790,0.237],
		[0.771,0.301], [0.678,0.203], [0.628,0.171], [0.615,0.208], [0.535,0.145], [0.730,0.262], [0.442,0.180], [0.605,0.218],
		[0.558,0.367], [0.800,0.220], [0.702,0.305], [0.675,0.162], [0.443,0.226], [0.659,0.311], [0.715,0.392], [0.556,0.107],
		[0.580,0.177], [0.635,0.272], [0.709,0.342], [0.640,0.237], [0.522,0.366], [0.416,0.284], [0.710,0.317], [0.598,0.166],
		[0.656,0.088], [0.641,0.204], [0.627,0.226], [0.627,0.139], [0.705,0.248], [0.709,0.290], [0.565,0.382], [0.485,0.410],
		[0.793,0.366], [0.680,0.371], [0.544,0.205], [0.711,0.306], [0.691,0.233], [0.693,0.283], [0.726,0.295], [0.915,0.283],
		[0.594,0.174], [0.641,0.179], [0.656,0.250], [0.706,0.365], [0.815,0.273], [0.648,0.311], [0.608,0.352], [0.760,0.314],
		[0.799,0.322], [0.640,0.235], [0.643,0.165], [0.581,0.353], [0.790,0.234], [0.656,0.406], [0.627,0.361], [0.730,0.364],
		[0.547,0.364], [0.519,0.358], [0.614,0.143], [0.493,0.135], [0.457,0.188], [0.768,0.354], [0.692,0.243], [0.677,0.164],
		[0.734,0.323], [0.819,0.288], [0.792,0.251], [0.670,0.284], [0.591,0.243], [0.769,0.191], [0.597,0.186], [0.704,0.301],
		[0.484,0.111], [0.618,0.156], [0.714,0.281], [0.695,0.351]
	];
	
	let data = new Dataset(gaussian_x);
    return data;
}

/**
    Aggregation dataset.
*/
function get_aggregation() {
    let agg_x = [
		[0.389,0.716], [0.361,0.709], [0.344,0.701], [0.325,0.729], [0.340,0.662], [0.310,0.696], [0.305,0.716], [0.324,0.649],
		[0.296,0.675], [0.279,0.718], [0.269,0.693], [0.241,0.711], [0.269,0.664], [0.290,0.647], [0.315,0.601], [0.277,0.630],
		[0.251,0.649], [0.233,0.681], [0.188,0.706], [0.212,0.676], [0.196,0.670], [0.170,0.671], [0.189,0.657], [0.225,0.646],
		[0.235,0.639], [0.221,0.615], [0.264,0.609], [0.264,0.589], [0.230,0.597], [0.184,0.619], [0.165,0.644], [0.145,0.674],
		[0.133,0.653], [0.135,0.631], [0.120,0.626], [0.160,0.620], [0.185,0.606], [0.107,0.600], [0.084,0.583], [0.107,0.569],
		[0.148,0.589], [0.171,0.581], [0.174,0.564], [0.138,0.565], [0.095,0.546], [0.104,0.509], [0.140,0.519], [0.160,0.549],
		[0.186,0.549], [0.194,0.530], [0.174,0.495], [0.151,0.505], [0.134,0.476], [0.158,0.478], [0.179,0.447], [0.205,0.501],
		[0.208,0.463], [0.226,0.455], [0.223,0.441], [0.251,0.430], [0.215,0.522], [0.216,0.547], [0.224,0.570], [0.224,0.555],
		[0.264,0.557], [0.284,0.586], [0.307,0.569], [0.279,0.551], [0.271,0.526], [0.246,0.517], [0.231,0.491], [0.267,0.509],
		[0.309,0.540], [0.321,0.519], [0.275,0.496], [0.247,0.466], [0.285,0.458], [0.309,0.470], [0.304,0.453], [0.299,0.431],
		[0.326,0.435], [0.338,0.466], [0.350,0.497], [0.388,0.429], [0.344,0.415], [0.195,0.342], [0.225,0.318], [0.201,0.323],
		[0.170,0.330], [0.155,0.314], [0.143,0.306], [0.129,0.284], [0.152,0.294], [0.176,0.311], [0.214,0.302], [0.177,0.299],
		[0.171,0.273], [0.148,0.258], [0.176,0.251], [0.191,0.277], [0.220,0.285], [0.225,0.273], [0.254,0.275], [0.273,0.250],
		[0.294,0.271], [0.276,0.227], [0.321,0.266], [0.340,0.277], [0.362,0.295], [0.425,0.323], [0.396,0.300], [0.406,0.292],
		[0.370,0.284], [0.344,0.261], [0.329,0.245], [0.295,0.224], [0.259,0.193], [0.283,0.199], [0.328,0.224], [0.352,0.251],
		[0.284,0.173], [0.310,0.177], [0.324,0.190], [0.334,0.206], [0.358,0.233], [0.378,0.256], [0.409,0.271], [0.406,0.255],
		[0.386,0.242], [0.381,0.216], [0.356,0.217], [0.376,0.195], [0.339,0.186], [0.349,0.168], [0.326,0.155], [0.270,0.146],
		[0.284,0.139], [0.310,0.145], [0.341,0.148], [0.328,0.128], [0.287,0.119], [0.310,0.109], [0.316,0.092], [0.348,0.124],
		[0.339,0.079], [0.352,0.102], [0.359,0.144], [0.379,0.177], [0.361,0.060], [0.375,0.085], [0.367,0.102], [0.381,0.068],
		[0.399,0.070], [0.398,0.100], [0.389,0.126], [0.379,0.149], [0.392,0.159], [0.409,0.134], [0.414,0.105], [0.426,0.128],
		[0.432,0.104], [0.426,0.092], [0.416,0.070], [0.430,0.051], [0.451,0.061], [0.465,0.086], [0.471,0.080], [0.486,0.066],
		[0.497,0.051], [0.495,0.081], [0.466,0.105], [0.466,0.119], [0.478,0.114], [0.441,0.143], [0.435,0.163], [0.392,0.179],
		[0.415,0.199], [0.517,0.086], [0.544,0.066], [0.555,0.087], [0.528,0.101], [0.520,0.118], [0.494,0.126], [0.510,0.141],
		[0.467,0.144], [0.460,0.150], [0.441,0.176], [0.466,0.182], [0.446,0.194], [0.429,0.215], [0.410,0.217], [0.401,0.240],
		[0.431,0.240], [0.445,0.233], [0.470,0.202], [0.485,0.190], [0.501,0.174], [0.501,0.159], [0.541,0.121], [0.576,0.084],
		[0.579,0.110], [0.554,0.130], [0.588,0.126], [0.575,0.144], [0.546,0.155], [0.525,0.179], [0.500,0.205], [0.471,0.226],
		[0.465,0.250], [0.434,0.271], [0.463,0.264], [0.439,0.283], [0.458,0.300], [0.475,0.291], [0.486,0.264], [0.503,0.235],
		[0.501,0.255], [0.480,0.306], [0.515,0.279], [0.532,0.291], [0.546,0.267], [0.524,0.255], [0.541,0.236], [0.519,0.219],
		[0.528,0.200], [0.544,0.205], [0.550,0.169], [0.566,0.166], [0.575,0.184], [0.555,0.217], [0.559,0.230], [0.560,0.251],
		[0.583,0.246], [0.591,0.227], [0.606,0.206], [0.588,0.196], [0.599,0.173], [0.590,0.143], [0.620,0.160], [0.826,0.096],
		[0.797,0.110], [0.760,0.141], [0.750,0.168], [0.738,0.204], [0.771,0.184], [0.794,0.149], [0.820,0.150], [0.820,0.120],
		[0.841,0.115], [0.840,0.136], [0.872,0.116], [0.865,0.101], [0.907,0.130], [0.899,0.151], [0.843,0.154], [0.843,0.176],
		[0.807,0.191], [0.799,0.204], [0.759,0.221], [0.767,0.229], [0.761,0.249], [0.795,0.236], [0.839,0.215], [0.868,0.200],
		[0.875,0.170], [0.903,0.188], [0.900,0.205], [0.914,0.216], [0.887,0.227], [0.881,0.235], [0.838,0.233], [0.812,0.241],
		[0.832,0.258], [0.765,0.263], [0.772,0.286], [0.760,0.301], [0.799,0.284], [0.824,0.279], [0.806,0.306], [0.782,0.318],
		[0.819,0.328], [0.828,0.319], [0.857,0.294], [0.866,0.275], [0.891,0.246], [0.889,0.269], [0.880,0.294], [0.874,0.319],
		[0.851,0.326], [0.830,0.354], [0.825,0.379], [0.815,0.404], [0.816,0.426], [0.794,0.430], [0.775,0.438], [0.761,0.451],
		[0.764,0.470], [0.756,0.485], [0.729,0.514], [0.767,0.501], [0.789,0.491], [0.801,0.465], [0.815,0.459], [0.836,0.436],
		[0.856,0.434], [0.846,0.460], [0.831,0.474], [0.807,0.496], [0.794,0.506], [0.802,0.529], [0.766,0.532], [0.738,0.540],
		[0.776,0.550], [0.774,0.566], [0.740,0.579], [0.730,0.596], [0.774,0.604], [0.799,0.579], [0.815,0.564], [0.841,0.547],
		[0.828,0.519], [0.846,0.500], [0.871,0.471], [0.885,0.484], [0.876,0.500], [0.909,0.515], [0.861,0.516], [0.875,0.526],
		[0.876,0.537], [0.855,0.544], [0.893,0.557], [0.896,0.581], [0.886,0.603], [0.855,0.573], [0.831,0.584], [0.840,0.597],
		[0.856,0.603], [0.879,0.633], [0.843,0.621], [0.812,0.617], [0.789,0.630], [0.756,0.608], [0.740,0.637], [0.762,0.637],
		[0.781,0.651], [0.769,0.672], [0.781,0.696], [0.818,0.705], [0.810,0.677], [0.809,0.649], [0.846,0.651], [0.841,0.675],
		[0.855,0.699], [0.881,0.650], [0.860,0.640], [0.504,0.522], [0.478,0.546], [0.469,0.574], [0.489,0.556], [0.512,0.546],
		[0.542,0.547], [0.525,0.565], [0.512,0.571], [0.480,0.593], [0.516,0.596], [0.492,0.615], [0.554,0.628], [0.542,0.595],
		[0.564,0.588], [0.583,0.611], [0.595,0.631], [0.573,0.580], [0.555,0.560], [0.571,0.547], [0.579,0.565], [0.617,0.555],
		[0.604,0.583], [0.130,0.054], [0.169,0.057], [0.135,0.068], [0.121,0.084], [0.143,0.086], [0.155,0.080], [0.180,0.069],
		[0.169,0.089], [0.145,0.100], [0.128,0.109], [0.136,0.121], [0.164,0.126], [0.155,0.106], [0.196,0.113], [0.181,0.089],
		[0.201,0.069], [0.202,0.089]
    ];
    
    let data = new Dataset(agg_x);
    return data;
}

/**
    Returns the dataset with the specified name.
*/
function get_dataset(name) {
    if (name == "spiral") {
        return get_spiral();
    }
    else if (name == "flame") {
        return get_flame();
    }
    else if (name == "moons") {
        return get_moons();
    }
    else if (name == "gaussian") {
        return get_gaussian();
    }
    else if (name == "aggregation") {
        return get_aggregation();
    }
    else if (name == "circle") {
        return get_circle();
    }
    else {
        throw("Unknown dataset: " + name);
    }
}

/**
    Returns the render options for the specified dataset.
*/
function get_render_options(name) {
    if (name == "spiral") {
        return [2.2, 2.2, -1.1, -1.1];
    }
    else if (name == "flame") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "moons") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "circle") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "gaussian") {
        return [1.1, 1.1, -0.05, -0.05];
    }
    else if (name == "aggregation") {
        return [1.0, 1.0, 0, -0.1];
    }
    else {
        throw("Unknown dataset: " + name);
    }
}

/**
	Returns the size of rendered markers.
*/
function get_marker_size(type) {
	if (type == "kmeans") {
		return 9;
	}
	else if (type == "dbscan") {
		return 5;
	}
	else if (type == "meanshift") {
		return 5;
	}
	return 5;
}

/**
	Returns the default hyperparameter settings for classifier and datasets combinations.
*/
function get_settings(type, name) {
    if (type == "kmeans") {
    	if (name == "spiral") {
	        return [3, 42];
	    }
	    else if (name == "flame") {
	        return [2, 42];
	    }
	    else if (name == "moons") {
	        return [2, 42];
	    }
	    else if (name == "circle") {
	        return [2, 42];
	    }
	    else if (name == "gaussian") {
	        return [4, 42];
	    }
	    else if (name == "aggregation") {
	        return [7, 42];
	    }
	    else {
        	return [3, 42];
        }
    }
    else if (type == "dbscan") {
    	if (name == "spiral") {
	        return [0.12, 6];
	    }
	    else if (name == "flame") {
	        return [0.06, 10];
	    }
	    else if (name == "moons") {
	        return [0.05, 6];
	    }
	    else if (name == "gaussian") {
	        return [0.05, 6];
	    }
	    else if (name == "aggregation") {
	        return [0.05, 6];
	    }
    	return [0.05, 6];
    }
    else if (type == "meanshift") {
    	if (name == "spiral") {
	        return [0.4];
	    }
	    else if (name == "gaussian") {
	        return [0.2];
	    }
    	return [0.1];
    }
}

/** ------------------------------------------------------

Utility functions and classes for classifiers.

--------------------------------------------------------- */

/**
    Holds a loaded and converted dataset.
*/
class Dataset {
    // Constructor
    constructor(x) {
        this.x = x;
    }

    // Returns a copy of this dataset
    clone() {
    	let nx = [];

    	// Copy all instances
    	for (let i = 0; i < this.no_examples(); i++) {
    		// Copy attributes of current instance
    		let e = this.x[i];
    		let new_e = [];
    		for (let a = 0; a < e.length; a++) {
    			new_e.push(e[a]);
    		}

    		// Add to new data arrays
    		nx.push(new_e);
    	}

    	// Create new dataset
    	let new_data = new Dataset(nx);
        return new_data;
    }

    // Returns thenumber of examples (size) of this dataset.
    no_examples() {
        return this.x.length;
    }

    // Returns the number of attributes in this dataset.
    no_attr() {
        return this.x[0].length;
    }

    // Feature-wise normalization where we subtract the mean and divide with stdev
    normalize() {
        for (let c = 0; c < this.no_attr(); c++) {
            let m = this.attr_mean(c); // mean
            let std = this.attr_stdev(c, m); // stdev
            for (let i = 0; i < this.no_examples(); i++) {
                this.x[i][c] = (this.x[i][c] - m) / std;
            }
        }
    }

    // Randomly shuffles the dataset examples
    shuffle() {
        // Create new arrays
        let nx = [];
        // Holds which instances that have been copied or not
        let done = new Array(this.x.length).fill(0);
        
        // Continue until all ínstances have been copied
        while (nx.length < this.x.length) {
            // Find a random instance that has not been copied
            let i = -1;
            while (i == -1) {
                let ti = Math.floor(rnd() * this.x.length);
                if (done[ti] == 0) {
                    // Not copied. Use this index.
                    done[ti] = 1;
                    i = ti;
                }
                else {
                    // Already copied. Get new index.
                    i = -1;
                }
            }
            
            // Get values
            let xv = this.x[i];
            
            // Copy to new arrays
            nx.push(xv);
        }

        this.x = nx;
    }

    // Calculates the mean value of an attribute
    attr_mean(c) {
        let m = 0;
        for (let i = 0; i < this.no_examples(); i++) {
            m += this.x[i][c];
        }
        m /= this.no_examples();
        return m;
    }

    // Calculates the min value of an attribute
    attr_min(c) {
        let m = 100000;
        for (let i = 0; i < this.no_examples(); i++) {
        	if (this.x[i][c] < m) {
        		m = this.x[i][c];
        	}
        }
        return m;
    }

    // Calculates the max value of an attribute
    attr_max(c) {
        let m = -100000;
        for (let i = 0; i < this.no_examples(); i++) {
        	if (this.x[i][c] > m) {
        		m = this.x[i][c];
        	}
        }
        return m;
    }

    // Calculates the standard deviation of an attribute
    attr_stdev(c, m) {
        let std = 0;
        for (let i = 0; i < this.no_examples(); i++) {
            std += Math.pow(this.x[i][c]  - m, 2);
        }
        std = Math.sqrt(std / (this.no_examples() - 1));
        return std;
    }
}
