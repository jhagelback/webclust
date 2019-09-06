
/** ------------------------------------------------------

Functions for rendering dataset and decision boundaries.

--------------------------------------------------------- */

// Init variables
var cellw = 5;
var canvas;
var ctx;

/**
    Clears the drawing canvas.
*/
function clear() {
    document.getElementById("clusters").innerHTML = "&nbsp;";
    document.getElementById("citer").innerHTML = "&nbsp;";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
    Inits the drawing canvas.
*/
function init() {
    canvas = document.getElementById('can');
    ctx = canvas.getContext("2d");
}

/**
    Draws the decision boundaries map.
*/
function draw_map(map) {
    for (let i = 0; i < map.length; i++) {
        let x1 = parseInt(i % 100);
        let x2 = parseInt(i / 100);
        drawcell(x1, x2, map[i]);
    }
}

/*
    Draws a filled cell with a different color for each class value.
*/
function drawcell(x, y, val) {
    let x1 = x * cellw;
    let y1 = y * cellw;
    
    // Select color based on class value
    let color = "#FFFFFF";
    if (val == 1) color = "#FFDBC2"; // red
    if (val == 2) color = "#C2DBFF"; // blue
    if (val == 3) color = "#DBFFC2"; // green
    if (val == 4) color = "#FAFC83"; // yellow
    if (val == 5) color = "#F7D4F3"; // pink
    if (val == 6) color = "#CCCCCC"; // gray
    if (val == 7) color = "#C3F4F3"; // cyan
    if (val == 8) color = "#CE6BF2"; // purple

    if (val == 9) color = "#DDB9A0"; // darker red
    if (val == 10) color = "#A0B9DD"; // darker blue
    if (val == 11) color = "#B9DDA0"; // darker green
    if (val == 12) color = "#D8DA61"; // darker yellow
    if (val == 13) color = "#D5B2D1"; // darker pink
    if (val == 14) color = "#AAAAAA"; // darker gray
    if (val == 15) color = "#A1D2D1"; // darker cyan
    if (val == 16) color = "#AC49D0"; // darker purple

    if (val == 17) color = "#BB9780"; // dark red
    if (val == 18) color = "#8097BB"; // dark blue
    if (val == 19) color = "#97BB80"; // dark green
    if (val == 20) color = "#B6B840"; // dark yellow
    if (val == 21) color = "#B390B0"; // dark pink
    if (val == 22) color = "#888888"; // dark gray
    if (val == 23) color = "#80B0B0"; // dark cyan
    if (val == 24) color = "#8A27B0"; // dark purple

    if (val == 25) color = "#666666"; // grey
    if (val == 26) color = "#FF6666"; // red
    if (val == 27) color = "#66FF66"; // green
    if (val == 28) color = "#6666FF"; // blue
    if (val == 29) color = "#444444"; // darker grey
    if (val == 30) color = "#DD4444"; // darker red
    if (val == 31) color = "#44DD44"; // darker green
    if (val == 32) color = "#4444DD"; // darker blue

    if (val > 32) color = "#EEEEEE"; // default color if more clusters than 32
    
    // Draw square
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1, cellw, cellw);
    ctx.closePath();
}

/**
    Draws the data points from the dataset.
*/
function draw_labels(data, opt) {
    for (let i = 0; i < data.no_examples(); i++) {
        let xe = data.x[i];
        let x1 = (xe[0] - opt[2]) / opt[0] * 100;
        let x2 = (xe[1] - opt[3]) / opt[1] * 100;
        draw_label(x1, x2);
    }
}

/**
    Draws a single data point from the dataset.
*/
function draw_label(x, y) {
    let x1 = x * cellw;
    let y1 = y * cellw;
    
    // Draw filled circle
    let r = cellw / 2;
    let c = cellw / 2;
    ctx.beginPath();
    ctx.fillStyle = "#000000";
    ctx.arc(x1 + c, y1 + c, r, 0, 2 * Math.PI, false);
    ctx.fill();
    ctx.closePath();
}


/**
    Draws markers, for example clusters in K-Means.
*/
function draw_markers(m, opt, msize) {
    for (let i = 0; i < m.length; i++) {
        let xe = m[i];
        let x1 = (xe[0] - opt[2]) / opt[0] * 100;
        let x2 = (xe[1] - opt[3]) / opt[1] * 100;
        draw_marker(x1, x2, msize);
    }
}

/**
    Draws a single marker.
*/
function draw_marker(x, y, w) {
    let x1 = x * cellw;
    let y1 = y * cellw;
    
    // Draw rectangle border
    let r = w / 2;
    ctx.beginPath();
    ctx.fillStyle = "#801103";
    ctx.rect(x1, y1, w, w);
    ctx.fill();
    ctx.closePath();

    // Draw filled rectangle
    r = w / 2 - 1;
    ctx.beginPath();
    ctx.fillStyle = "#f0351d";
    ctx.rect(x1 + 1, y1 + 1, w - 2, w - 2);
    ctx.fill();
    ctx.closePath();
}