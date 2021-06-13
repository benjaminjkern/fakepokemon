let canvas;
const controlVars = { kill: false, killed: true };
const settings = { screenSize: 0, pixelSize: 32, frame: 1, priority: 1 };
const keysDown = {};

let best;
let brain;
let showing;
let targets;
let mutate;
let count = 0;

window.onload = function() {
    canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');

    window.onresize = function() {
        start(ctx);
    }

    window.onkeydown = function(e) {
        if (e.code === 'ArrowUp') {
            settings.priority = (settings.priority + 1) % 3;
            restart(ctx);
        } else if (e.code === 'ArrowDown') {
            settings.priority = (settings.priority + 2) % 3;
            restart(ctx);
        } else if (e.code === 'ArrowLeft') {
            settings.pixelSize = Math.max(1, settings.pixelSize / 2);
            start(ctx);
        } else if (e.code === 'ArrowRight') {
            settings.pixelSize *= 2;
            start(ctx);
        }
        keysDown[e.code] = true;
    }

    window.onkeyup = function(e) {
        delete keysDown[e.code];
    }

    window.onmousedown = function(e) {
        const dx = (2 * e.x / canvas.width - 1) * settings.frame;
        const dy = (2 * e.y / canvas.height - 1) * settings.frame;
        console.log(dx, dy);
        console.log(best.pass([dx, dy]));
    }

    restart(ctx);

    drawLoop(ctx);
};

const restart = (ctx) => {
    best = undefined;
    brain = newNeuralNet([2, 100, 3], 1);
    targets = Array(100).fill().map(() => ({
        pos: [initRand(settings.frame), initRand(settings.frame)],
        color: Array(3).fill().map(() => Math.random()),
    }));
    mutate = 0.01;

    start(ctx);
}


const start = (ctx) => {
    controlVars.kill = true;
    if (!controlVars.killed) return;
    controlVars.kill = false;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    settings.screenSize = Math.min(canvas.width, canvas.height);

    if (!best && brain) best = brain;
    // console.log(best);

    initframe(ctx);
    makeframe(ctx);
}

const drawLoop = (ctx) => {
    if (count > 0) setTimeout(() => drawLoop(ctx), 1000);
    console.log("Tested: " + count + "\nBest: " + best.min + "\nAverage: " + best.average + "\nWorst: " + best.max);
    count = 0;
    draw(ctx);
}

const initframe = (ctx) => {}

const makeframe = (ctx) => {
    count++;
    if (controlVars.kill) {
        controlVars.killed = true;
        setTimeout(() => start(ctx), 1000);
        return;
    }
    controlVars.killed = false;

    brain.errors = targets.map(target => brain.error(target.pos, target.color));

    brain.min = Math.min(...brain.errors);
    brain.max = Math.max(...brain.errors);
    brain.average = brain.errors.reduce((p, c) => p + c, 0) / targets.length;

    switch (settings.priority) {
        case 0: // Best
            brain.errorScore = brain.errors.reduce((p, c) => c > brain.min ? Math.min(p, c) : p, brain.max);
            break;
        case 1: // Average
            brain.errorScore = brain.average;
            break;
        case 2: // Worst
            brain.errorScore = brain.max;
            break;
    }
    if (!best || brain.errorScore < best.errorScore) {
        // best = brain;
    }

    brain = mutateNeuralNet(best, 0.01);
    brain.backPropMulti(targets.map(target => target.pos), targets.map(target => target.color), 1);

    setTimeout(() => {
        makeframe(ctx);
    }, 1);
}

const draw = (ctx) => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    for (let y = 0; y < Math.ceil(canvas.height / settings.pixelSize); y++) {
        for (let x = 0; x < Math.ceil(canvas.width / settings.pixelSize); x++) {
            const dx = (2 * x * settings.pixelSize / canvas.width - 1) * settings.frame;
            const dy = (2 * y * settings.pixelSize / canvas.height - 1) * settings.frame;
            const output = best.pass([dx, dy]).flat();
            const alpha = (output[3] || 1) * 255 * Math.min(1, settings.pixelSize ** 2);
            for (let py = 0; py < settings.pixelSize; py++) {
                if (Math.floor(y * settings.pixelSize + py) < 0 || Math.floor(y * settings.pixelSize + py) >= canvas.height) continue;
                for (let px = 0; px < settings.pixelSize; px++) {
                    if (Math.floor(x * settings.pixelSize + px) < 0 || Math.floor(x * settings.pixelSize + px) >= canvas.width) continue;
                    const j = 4 * (Math.floor(x * settings.pixelSize + px) + canvas.width * Math.floor(y * settings.pixelSize + py));
                    for (let k = 0; k < 3; k++) {
                        data[j + k] += output[k] * alpha;
                    }
                }
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);

    for (const target of targets) {
        ctx.fillStyle = `rgb(${target.color.map(c => Math.floor(c * 255)).join(',')})`;
        ctx.strokeStyle = "black";
        ctx.beginPath();
        ctx.arc((target.pos[0] / settings.frame + 1) * canvas.width / 2, (target.pos[1] / settings.frame + 1) * canvas.height / 2, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    }
}





/*
UTIL FUNCTIONS

*/


const addEventListener = (target, listenerType, func) => {
    if (!target[listenerType]) target[listenerType] = func;
    else target[listenerType] = function(e) {
        target[listenerType](e);
        func(e);
    }
};