let canvas;
const controlVars = { kill: false, killed: true };
const settings = {};
const keysDown = {};

let generator;
let autoencoder;
let fakeOutput;
let loading = false;
let loaded = false;
let real;
let inputImage;
let fakeImage;
let count = 0;
const image = new Image(96, 96);

let pokemon = [];

window.onload = function() {
    canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');

    // window.onresize = function() {
    //     start(ctx);
    // }

    window.onkeydown = function(e) {
        keysDown[e.code] = true;
    }

    window.onkeyup = function(e) {
        delete keysDown[e.code];
    }

    restart(ctx);
    // drawLoop(ctx);
};

const restart = (ctx) => {
    autoencoder = newNeuralNet([96 * 96 * 4, 100, 96 * 96 * 4], 1);

    start(ctx);
    drawLoop(ctx);
}


const start = (ctx) => {
    controlVars.kill = true;
    if (!controlVars.killed) return;
    controlVars.kill = false;

    canvas.width = 96 * 2;
    canvas.height = 96;

    makeframe(ctx);
}

const drawLoop = (ctx) => {
    setTimeout(() => drawLoop(ctx), 1000);
    if (count === 0 || !fakeImage) return;

    count = 0;
    ctx.clearRect(96, 0, 96, 96);
    const imageData = ctx.getImageData(96, 0, 96, 96);
    fakeImage.forEach((o, i) => imageData.data[i] = Math.floor(o * 256));
    ctx.putImageData(imageData, 96, 0);
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

    if (!loading) {
        if (!loaded) {
            loading = true;
            image.src = 'black-white/' + Math.floor(Math.random() * 649 + 1) + '.png';

            image.onload = () => {
                loading = false;
                loaded = true;
                ctx.clearRect(0, 0, 96, 96);

                ctx.drawImage(image, 0, 0);
                const imageData = ctx.getImageData(0, 0, 96, 96);
                inputImage = [...imageData.data].map(d => d / 255);
            }
        }
        if (loaded) {
            pokemon.push(inputImage);
            loaded = false;
        }
    }
    if (pokemon.length === 10) {

        autoencoder.count = (autoencoder.count || 0) + 1;
        let factor = 1;

        autoencoder.totalWeight = (autoencoder.totalWeight || 0) + factor;
        autoencoder.totalError = (autoencoder.totalError || 0) + pokemon.reduce((acc, pok) => acc + autoencoder.error(pok, pok), 0) * factor / pokemon.length;

        console.log(autoencoder.totalError / autoencoder.totalWeight);

        autoencoder.backPropMulti(pokemon, pokemon, 1);

        // const generator = mutateNeuralNet(autoencoder, 0);
        // const mid = Math.floor(generator.layerCounts.length / 2);
        // generator.layers = generator.layers.slice(mid);
        // fakeImage = generator.pass(Array(generator.layerCounts[mid]).fill().map(()=>Math.random()));
        fakeImage = autoencoder.pass(inputImage);

        pokemon = [];
    }

    setTimeout(() => {
        makeframe(ctx);
    }, 1);
}