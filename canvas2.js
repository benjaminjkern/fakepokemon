let canvas;
const controlVars = { kill: false, killed: true };
const settings = {};
const keysDown = {};

const mutate = false;
const backprop = true;

let lowesterror = Number.MAX_VALUE;
let avgerror = 0;
let best;
let autoencoder;
let fakeOutput;
let loading = false;
let loaded = false;
let real;
let inputImage;

let fakeImages;
let autoEncodedImages;
let pickedPokemon;
let count = 0;
let iters = 0;
const image = new Image(96, 96);

let testPokemon = [];
let pokemon = [];

window.onload = function () {
    canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");

    // window.onresize = function() {
    //     start(ctx);
    // }

    window.onkeydown = function (e) {
        keysDown[e.code] = true;
    };

    window.onkeyup = function (e) {
        delete keysDown[e.code];
    };
    restart(ctx);
};

const restart = (ctx) => {
    autoencoder = new LinearNeuralNet([96 * 96 * 4, 1, 96 * 96 * 4]);
    // autoencoder = new LinearNeuralNet([
    //     96 * 96 * 4,
    //     100,
    //     60,
    //     30,
    //     10,
    //     30,
    //     60,
    //     100,
    //     96 * 96 * 4,
    // ]);

    // autoencoder = new ConvolutionalNeuralNet(
    //     1,
    //     [96, 96],
    //     [
    //         {
    //             channels: 3,
    //             kernelSpecs: { kernelSize: [9, 9], padding: 4, stride: 2 },
    //         },
    //         {
    //             channels: 3,
    //             kernelSpecs: { kernelSize: [3, 3], padding: 0, stride: 2 },
    //         },
    //         {
    //             channels: 5,
    //             kernelSpecs: { kernelSize: [5, 5], padding: 2, stride: 1 },
    //         },
    //         {
    //             channels: 5,
    //             kernelSpecs: { kernelSize: [3, 3], padding: 0, stride: 2 },
    //         },
    //         {
    //             channels: 8,
    //             kernelSpecs: { kernelSize: [3, 3], padding: 1, stride: 1 },
    //         },
    //         {
    //             channels: 8,
    //             kernelSpecs: { kernelSize: [3, 3], padding: 0, stride: 2 },
    //         },
    //         {
    //             channels: 10,
    //             kernelSpecs: { kernelSize: [5, 5], padding: 0, stride: 1 },
    //         },
    //         {
    //             channels: 8,
    //             kernelSpecs: {
    //                 kernelSize: [3, 3],
    //                 padding: 2,
    //                 innerPadding: 1,
    //             },
    //         },
    //         {
    //             channels: 8,
    //             kernelSpecs: {
    //                 kernelSize: [3, 3],
    //                 padding: 1,
    //                 innerPadding: 1,
    //             },
    //         },
    //         {
    //             channels: 5,
    //             kernelSpecs: {
    //                 kernelSize: [3, 3],
    //                 padding: 1,
    //                 innerPadding: 1,
    //             },
    //         },
    //         {
    //             channels: 5,
    //             kernelSpecs: {
    //                 kernelSize: [5, 5],
    //                 padding: 4,
    //                 innerPadding: 1,
    //             },
    //         },
    //         {
    //             channels: 3,
    //             kernelSpecs: {
    //                 kernelSize: [3, 3],
    //                 padding: 2,
    //                 innerPadding: 1,
    //             },
    //         },
    //         {
    //             channels: 3,
    //             kernelSpecs: {
    //                 kernelSize: [9, 9],
    //                 padding: 7,
    //                 innerPadding: 1,
    //             },
    //         },
    //         {
    //             channels: 1,
    //             kernelSpecs: {
    //                 kernelSize: [10, 10],
    //                 padding: 7,
    //                 innerPadding: 0,
    //             },
    //         },
    //     ]
    // );

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
};

const numPokemon = 2;

const batchSize = 2;

const start = (ctx) => {
    controlVars.kill = true;
    if (!controlVars.killed) return;
    controlVars.kill = false;

    canvas.width = 96 * batchSize;
    canvas.height = 96 * 3;

    loading = true;
    initframe(ctx);
    // makeframe(ctx);
};

const drawLoop = (ctx) => {
    setTimeout(() => drawLoop(ctx), 1000);
    if (count === 0 || !inputImage) return;

    count = 0;

    let imageData;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let encodedImage;
    for (let p = 0; p < batchSize; p++) {
        encodedImage = autoEncodedImages[p];
        imageData = ctx.getImageData(96 * p, 0, 96, 96);
        encodedImage.data.forEach(mapImage(imageData));
        ctx.putImageData(imageData, 96 * p, 0);

        encodedImage = pickedPokemon[p];
        imageData = ctx.getImageData(96 * p, 96, 96, 96);
        encodedImage.data.forEach(mapImage(imageData));
        ctx.putImageData(imageData, 96 * p, 96);

        encodedImage = fakeImages[p];
        imageData = ctx.getImageData(96 * p, 96 * 2, 96, 96);
        encodedImage.data.forEach(mapImage(imageData));
        ctx.putImageData(imageData, 96 * p, 96 * 2);
    }
};

const mapToBlackAndWhite = (imageData) => (o, i) => {
    imageData.data[4 * i] = Math.floor(o * 256);
    imageData.data[4 * i + 1] = Math.floor(o * 256);
    imageData.data[4 * i + 2] = Math.floor(o * 256);
    imageData.data[4 * i + 3] = 255;
};

const mapImage = (imageData) => (o, i) => {
    imageData.data[i] = Math.floor(o * 256);
};

const avgweight = (count) => Math.exp(1) * Math.log(count + 1);
const totalweight = (count) => Math.exp(1) * count * Math.log(count + 1);

const initframe = (ctx) => {
    if (loading) {
        if (!loaded) {
            image.src = "black-white/" + (pokemon.length + 1) + ".png";

            image.onload = () => {
                loaded = true;
                ctx.clearRect(0, 0, 96, 96);

                ctx.drawImage(image, 0, 0);
                const imageData = ctx.getImageData(0, 0, 96, 96);
                // inputImage = [];
                // for (let i = 0; i < 96; i++) {
                //     for (let j = 0; j < 96; j++) {
                //         const index = j + 96 * i;
                //         inputImage[index] = (imageData.data[4 * index] + imageData.data[4 * index + 1] + imageData.data[4 * index + 2]) / 255 / 3;
                //     }
                // }
                inputImage = [...imageData.data].map((d) => d / 255);
            };
        }
        if (loaded) {
            pokemon[pokemon.length] = newTensor([96 * 96 * 4], inputImage);
            // pokemon[pokemon.length] = newTensor([96, 96, 1], inputImage);
            pokemon[pokemon.length - 1].name = pokemon.length;
            loaded = false;
            if (pokemon.length >= numPokemon) loading = false;
        }

        setTimeout(() => initframe(ctx), 1);
    } else {
        testPokemon = pokemon.slice(numPokemon);
        pokemon = pokemon.slice(0, numPokemon);
        makeframe(ctx);
        drawLoop(ctx);
    }
};
previouslyPicked = {};

const makeframe = (ctx) => {
    count++;
    iters++;
    if (controlVars.kill) {
        controlVars.killed = true;
        if (controlVars.continue) setTimeout(() => start(ctx), 1000);
        return;
    }
    controlVars.killed = false;

    const validPokemon = pokemon.filter((pok) => !previouslyPicked[pok.name]);
    if (validPokemon.length >= batchSize) {
        pickedPokemon = pickUnique(validPokemon, batchSize);
    } else {
        previouslyPicked = {};
        for (pok of validPokemon) {
            previouslyPicked[pok.name] = true;
        }
        const validPokemon2 = pokemon.filter(
            (pok) => !previouslyPicked[pok.name]
        );
        pickedPokemon = [
            ...pickUnique(validPokemon, validPokemon.length),
            ...pickUnique(validPokemon2, batchSize - validPokemon.length),
        ];
    }
    for (pok of pickedPokemon) {
        previouslyPicked[pok.name] = true;
    }
    // pickedPokemon = pokemon;

    autoencoder.error(pickedPokemon, pickedPokemon);
    avgerror = (avgerror * (iters - 1) + autoencoder.lastError) / iters;

    const status = document.getElementById("status");
    const stats = document.getElementById("stats");
    const dateSpan = `<span style="color: gray; font-size: 0.75em"> - ${new Date().toLocaleString()}</span>`;
    if (autoencoder.lastError < lowesterror) {
        lowesterror = autoencoder.lastError;
        status.innerHTML = `<span style="color: red">${autoencoder.lastError}</span>${dateSpan}<br>${status.innerHTML}`;
    } else {
        status.innerHTML = `${autoencoder.lastError}${dateSpan}<br>${status.innerHTML}`;
    }
    stats.innerHTML = `<b>Best</b>: ${lowesterror}<br><b>Average</b>: ${avgerror}`;

    autoencoder.backProp(
        pickedPokemon,
        pickedPokemon,
        0.001,
        batchSize / numPokemon
    );
    // console.log(autoencoder);

    autoEncodedImages = autoencoder.pass(pickedPokemon);

    const decoder = autoencoder.copy();
    // const encoder = autoencoder.copy();
    const mid = Math.floor(decoder.layers.length / 2 + 1);
    decoder.layers = decoder.layers.slice(mid);
    // encoder.layers = encoder.layers.slice(0, mid - 1);
    const randomInput = Array(batchSize)
        .fill()
        .map(() => randomTensor(decoder.layers[0].inputSize));
    fakeImages = decoder.pass(randomInput);
    // console.log(encoder.pass(pickedPokemon).map((t) => t.data.join(",")));

    // const encoder = autoencoder.copy()
    // encoder.layers = encoder.layers.slice(0, mid);
    // console.log(autoencoder.layers[0].pass([pok]));
    // for (const pok of pokemon) {
    //     console.log(autoencoder.layers[0].pass([pok]));
    // }
    // console.log(encoder.pass(pokemon));

    setTimeout(() => {
        makeframe(ctx);
    }, 1);
};

window.onclick = () => {
    controlVars.kill = true;
    controlVars.continue = false;
};

const pickUnique = (list, count) => {
    if (count === 0) return [];
    const r = Math.floor(Math.random() * list.length);
    return [
        list[r],
        ...pickUnique(
            list.filter((_, i) => i !== r),
            count - 1
        ),
    ];
};
