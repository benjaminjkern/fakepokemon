let canvas;
const controlVars = { kill: false, killed: true };
const settings = { screenSize: 0, pixelSize: 64, frame: 1, priority: 1 };

let brain;
let targets;

let avgerror = 0;
let lowesterror = Number.MAX_SAFE_INTEGER;
let iters = 0;
paused = false;

window.onload = function () {
    canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');

    window.onmousedown = function (e) {
        const dx = (2 * e.x / canvas.width - 1) * settings.frame;
        const dy = (2 * e.y / canvas.height - 1) * settings.frame;
        console.log(dx, dy);
        const result = brain.pass([newTensor([2], [dx, dy])])[0].data;
        console.log(result);

        const width = 2 * 5 * settings.frame / canvas.width;

        for (const target of targets) {
            if ((dx - target.pos.data[0]) ** 2 + (dy - target.pos.data[1]) ** 2 < width ** 2) {
                console.log(target);
                break;
            }
        }
    }

    window.onkeydown = function (e) {
        if (paused) return;
        paused = true;
        settings.pixelSize = 4;
        draw(ctx);
    }

    window.onkeyup = function (e) {
        paused = false;
        settings.pixelSize = 64;
    }

    restart(ctx);

    drawLoop(ctx);
};

const randomColors = 2;
const randomColor = () => {
    const r = Math.floor(Math.random() * randomColors);
    const a = Array(randomColors).fill(0);
    a[r] = 1;
    return a;
}
const colors = Array(randomColors).fill().map(() => Array(3).fill().map(() => Math.random()));
const getColor = (output) => {
    let max = -Number.MAX_VALUE;
    let maxi = -1;
    for (const [j, o] of output.entries()) {
        if (o > max) {
            max = o;
            maxi = j;
        }
    }
    return colors[maxi];

    // const sum = output.reduce((p, c) => p + c, 0);

    // const color = [0, 0, 0];
    // for (const [j, o] of output.entries()) {
    //     for (let i = 0; i < 3; i++) {
    //         color[i] += o * colors[j][i] / sum;
    //     }
    // }
    // return color;
}

const restart = (ctx) => {
    brain = new LinearNeuralNet([2, 10, 10, 10, 10, 10, 10, 3], { batchNormalize: false, finalActivation: sigmoid });
    targets = Array(50).fill().map(() => ({
        pos: randomTensor([2], settings.frame),
        color: randomTensor([3], 0.5, 0.5),
    }));

    start(ctx);
}


const start = (ctx) => {
    controlVars.kill = true;
    if (!controlVars.killed) return;
    controlVars.kill = false;

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    settings.screenSize = Math.min(canvas.width, canvas.height);
    // console.log(best);

    initframe(ctx);
    makeframe(ctx);
}

const drawLoop = (ctx) => {
    if (iters > 0) setTimeout(() => drawLoop(ctx), 1000);
    if (!paused) draw(ctx);
}

const initframe = (ctx) => { }

const makeframe = (ctx) => {
    if (controlVars.kill) {
        controlVars.killed = true;
        setTimeout(() => start(ctx), 1000);
        return;
    }
    controlVars.killed = false;
    if (!paused) {
        iters++;

        brain.error(targets.map(target => target.pos), targets.map(target => target.color));

        avgerror = (avgerror * (iters - 1) + brain.lastError) / iters;

        if (brain.lastError < lowesterror) {
            lowesterror = brain.lastError;
        }

        const status = document.getElementById('status');
        status.innerHTML = `${brain.lastError}, ${lowesterror}`;

        brain.backProp(targets.map(target => target.pos), targets.map(target => target.color), 1);
    }

    setTimeout(() => {
        makeframe(ctx);
    }, 1);
}

const draw = (ctx) => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    const ysize = Math.ceil(canvas.height / settings.pixelSize);
    const xsize = Math.ceil(canvas.width / settings.pixelSize);

    const inputs = Array(xsize * ysize).fill().map((_, i) => {
        const [x, y] = [i % xsize, Math.floor(i / xsize)];
        const dx = (2 * (x + 0.5) * settings.pixelSize / canvas.width - 1) * settings.frame;
        const dy = (2 * (y + 0.5) * settings.pixelSize / canvas.height - 1) * settings.frame;
        return newTensor([2], [dx, dy]);
    });
    // console.log(inputs);

    const outputs = brain.pass(inputs);
    // console.log(inputs[0].data);
    // console.log(outputs[0].data);

    for (let y = 0; y < ysize; y++) {
        for (let x = 0; x < xsize; x++) {
            const output = outputs[y * xsize + x].data;
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
        ctx.fillStyle = `rgb(${target.color.data.map(c => Math.floor(c * 255)).join(',')})`;
        ctx.strokeStyle = "black";
        ctx.beginPath();
        ctx.arc((target.pos.data[0] / settings.frame + 1) * canvas.width / 2, (target.pos.data[1] / settings.frame + 1) * canvas.height / 2, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    }
}





/*
UTIL FUNCTIONS

*/


const addEventListener = (target, listenerType, func) => {
    if (!target[listenerType]) target[listenerType] = func;
    else target[listenerType] = function (e) {
        target[listenerType](e);
        func(e);
    }
};