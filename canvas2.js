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
    // autoencoder = newNeuralNet([96 * 96, 1000, 100, 10, 100, 1000, 96 * 96], 1);

    autoencoder = newConvolutionalNeuralNet(1, [96, 96], [
        { channels: 3, kernelSpecs: { kernelSize: [20, 20], padding: 0, stride: 5 } },
        { channels: 5, kernelSpecs: { kernelSize: [16, 16], padding: 0, stride: 2 }, },
        { channels: 3, kernelSpecs: { kernelSize: [16, 16], padding: 15, innerPadding: 2 } },
        { channels: 1, kernelSpecs: { kernelSize: [20, 20], padding: 12, innerPadding: 5 } },
    ]);

    // autoencoder.layers[3].bias[0] = 0;

    // autoencoder = newConvolutionalNeuralNet(10, [2, 2], [
    //     { channels: 1, kernelSpecs: { kernelSize: [50, 50], padding: 49, innerPadding: 45 } }
    // ]);
    // console.log(autoencoder);

    // const testNet = newConvolutionalNeuralNet(1, [10, 10], [
    //     { channels: 5, kernelSpecs: { kernelSize: [2, 2], padding: 4 } },
    //     { channels: 10, kernelSpecs: { kernelSize: [5, 5], padding: 2 } },
    //     { channels: 15, kernelSpecs: { kernelSize: [5, 5], padding: } },
    //     { channels: 20, kernelSpecs: { kernelSize: [3, 3] } },
    // ]);
    // const input = randomTensor([10, 10, 1]);
    // console.log(testNet.pass(input));

    // const testNet = newConvolutionalNeuralNet(1, [2, 2], [
    //     { channels: 1, kernelSpecs: { kernelSize: [2, 2], padding: 1 } },
    // ]);
    // const input = randomTensor([2, 2, 1]);
    // const dotProduct = sumOverIndices([2, 2, 1], ([m, n, i]) => input.get([m, n, i]) * testNet.layers[0].kernels[0][0].kernel.get([m, n]));
    // console.log(dotProduct + testNet.layers[0].bias[0]);
    // console.log(testNet.pass(input));

    start(ctx);
}

const numPokemon = 3;


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
        encodedImage.data.forEach(mapToBlackAndWhite(imageData));
        ctx.putImageData(imageData, 96 * p, 0);
    }
    imageData = ctx.getImageData(0, 96, 96, 96);
    fakeImage.data.forEach(mapToBlackAndWhite(imageData));
    ctx.putImageData(imageData, 0, 96);

    imageData = ctx.getImageData(96, 96, 96, 96);
    realImage.data.forEach(mapToBlackAndWhite(imageData));
    ctx.putImageData(imageData, 96, 96);

    imageData = ctx.getImageData(96 * 2, 96, 96, 96);
    realFakeImage.data.forEach(mapToBlackAndWhite(imageData));
    ctx.putImageData(imageData, 96 * 2, 96);
}

const mapToBlackAndWhite = (imageData) => (o, i) => {
    imageData.data[4 * i] = Math.floor(o * 256)
    imageData.data[4 * i + 1] = Math.floor(o * 256)
    imageData.data[4 * i + 2] = Math.floor(o * 256)
    imageData.data[4 * i + 3] = 255;
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
                inputImage = [];
                for (let i = 0; i < 96; i++) {
                    for (let j = 0; j < 96; j++) {
                        const index = j + 96 * i;
                        inputImage[index] = (imageData.data[4 * index] + imageData.data[4 * index + 1] + imageData.data[4 * index + 2]) / 255 / 3;
                    }
                }
                // inputImage = [...imageData.data].map(d => d / 255);
            }
        }
        if (loaded) {
            pokemon[pokemon.length] = newTensor([96, 96, 1], inputImage);
            pokemon[pokemon.length - 1].name = pokemon.length;
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

    autoencoder.error(pokemon, pokemon);
    console.log(autoencoder.lastError);

    autoencoder.backPropMulti(pokemon, pokemon, 0.1);
    // console.log(autoencoder);

    fakeImages = pokemon.map(pok => autoencoder.pass(pok));

    realImage = testPokemon[Math.floor(Math.random() * testPokemon.length)];

    realFakeImage = autoencoder.pass(realImage);

    const decoder = {...autoencoder };
    const mid = Math.floor(decoder.layers.length / 2);
    decoder.layers = decoder.layers.slice(mid);
    const randomInput = randomTensor([1, 1, decoder.layers[0].inputChannels])
    fakeImage = decoder.pass(randomInput);

    // const randomInput = randomTensor([...autoencoder.layers[0].kernelSpecs.inputSize, autoencoder.layers[0].inputChannels]);
    // console.log(randomInput);
    // fakeImage = autoencoder.pass(randomInput);

    // const encoder = mutateNeuralNet(autoencoder, 0);
    // encoder.layers = encoder.layers.slice(0, mid);
    // for (const poke of pokemon) {
    //     console.log(encoder.pass(poke));
    // }

    setTimeout(() => {
        makeframe(ctx);
    }, 1);
}