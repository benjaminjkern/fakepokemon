let canvas;
const controlVars = { kill: false, killed: true };
const settings = {};
const keysDown = {};

const mutate = false;
const backprop = true;

let best;
let autoencoder;
let fakeOutput;
let loading = false;
let loaded = false;
let real;
let inputImage;
let fakeImages;
let fakeImage;
let realImage;
let realFakeImage;
let count = 0;
const image = new Image(96, 96);

let testPokemon = [];
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
};

const restart = (ctx) => {
    autoencoder = newNeuralNet([96 * 96 * 4, 100, 10, 100, 96 * 96 * 4], 1);

    start(ctx);
}

const numPokemon = 2;


const start = (ctx) => {
    controlVars.kill = true;
    if (!controlVars.killed) return;
    controlVars.kill = false;

    canvas.width = 96 * numPokemon;
    canvas.height = 96 * 2;

    loading = true;
    initframe(ctx);
    // makeframe(ctx);
}

const drawLoop = (ctx) => {
    setTimeout(() => drawLoop(ctx), 1000);
    if (count === 0 || !inputImage) return;

    count = 0;

    let imageData;

    ctx.clearRect(0, 0, 96 * (numPokemon), 96 * 2);
    for (let p = 0; p < numPokemon; p++) {
        const encodedImage = fakeImages[p];
        imageData = ctx.getImageData(96 * p, 0, 96, 96);
        encodedImage.forEach((o, i) => imageData.data[i] = Math.floor(o * 256));
        ctx.putImageData(imageData, 96 * p, 0);
    }
    imageData = ctx.getImageData(0, 96, 96, 96);
    fakeImage.forEach((o, i) => imageData.data[i] = Math.floor(o * 256));
    ctx.putImageData(imageData, 0, 96);

    imageData = ctx.getImageData(96, 96, 96, 96);
    realImage.forEach((o, i) => imageData.data[i] = Math.floor(o * 256));
    ctx.putImageData(imageData, 96, 96);

    imageData = ctx.getImageData(96 * 2, 96, 96, 96);
    realFakeImage.forEach((o, i) => imageData.data[i] = Math.floor(o * 256));
    ctx.putImageData(imageData, 96 * 2, 96);
}

const initframe = (ctx) => {
    if (loading) {
        if (!loaded) {
            image.src = 'black-white/' + (pokemon.length + 1) + '.png';

            image.onload = () => {
                loaded = true;
                ctx.clearRect(0, 0, 96, 96);

                ctx.drawImage(image, 0, 0);
                const imageData = ctx.getImageData(0, 0, 96, 96);
                inputImage = [...imageData.data].map(d => d / 255);
            }
        }
        if (loaded) {
            pokemon[pokemon.length] = inputImage;
            loaded = false;
            if (pokemon.length >= numPokemon * 2)
                loading = false;
        }

        setTimeout(() => initframe(ctx), 1);
    } else {
        testPokemon = pokemon.slice(numPokemon);
        pokemon = pokemon.slice(0, numPokemon);
        makeframe(ctx);
        drawLoop(ctx);
    }
}

const makeframe = (ctx) => {
    count++;
    if (controlVars.kill) {
        controlVars.killed = true;
        setTimeout(() => start(ctx), 1000);
        return;
    }
    controlVars.killed = false;


    autoencoder.totalError = autoencoder.error(pokemon, pokemon);
    console.log(autoencoder.totalError);

    if (backprop) {
        autoencoder.backPropMulti(pokemon, pokemon, 0.1);
    }
    if (mutate) {
        if (!best || best.totalError > autoencoder.totalError) {
            console.log("%cYEE", "color:red");
            best = autoencoder;
        }

        autoencoder = mutateNeuralNet(best, 0.1);
    }

    fakeImages = pokemon.map(pok => (best || autoencoder).pass(pok));

    realImage = testPokemon[Math.floor(Math.random() * testPokemon.length)];

    realFakeImage = autoencoder.pass(realImage);

    const decoder = mutateNeuralNet(autoencoder, 0);
    const encoder = mutateNeuralNet(autoencoder, 0);
    const mid = Math.floor(decoder.layerCounts.length / 2);
    decoder.layers = decoder.layers.slice(mid);
    encoder.layers = encoder.layers.slice(0, mid);
    fakeImage = decoder.pass(Array(decoder.layerCounts[mid]).fill().map(() => Math.random()));

    // for (const poke of pokemon) {
    //     console.log(encoder.pass(poke));
    // }

    setTimeout(() => {
        makeframe(ctx);
    }, 1);
}