const mutateNeuralNet = (oldNeuralNet, mutation = 1) => {
    const newNet = newNeuralNet(oldNeuralNet.layerCounts, mutation);
    newNet.layers.forEach((layer, i) => {
        layer.weights = matadd(layer.weights, oldNeuralNet.layers[i].weights);
        layer.bias = matadd(layer.bias, oldNeuralNet.layers[i].bias);
    })
    return newNet;
}

const newNeuralNet = (layerCounts, randomRange = 1) => ({
    name: Math.random().toString(36).substring(2),
    layers: layerCounts.slice(1).map((layerCount, i) => newLinearLayer(layerCounts[i], layerCount, randomRange, ...(i === layerCounts.length - 2 ? [] : [ReLU, dReLU]))),
    // layerCounts,
    pass(input) {
        return this.layers.reduce((p, layer) => layer.pass(p), input.map(i => [i])).flat();
    },
    error(inputs, targets) {
        return inputs.reduce((acc, input, t) => {
            const target = targets[t];
            return acc + this.pass(input).reduce((p, c, i) => p + (c - target[i]) ** 2, 0);
        }, 0) * 0.5 / inputs.length;
    },
    backPropMulti(inputs, targets, alpha) {
        inputs.forEach((input, j) => {
            // TODO: derror

            // THIS IS CROSS ENTROPY DONT USE IT IT BROKEN
            // let A = this.pass(input).map((y, i) => targets.reduce((p, target, t) => p +
            //         (t === j ? y - target[i] : ((y + target[i] - 1) * (targets[t][i] - targets[j][i]) ** 2)), 0) *
            //     this.layers[this.layers.length - 1].dsigma(y));
            let A = this.pass(input).map((y, i) => y - targets[j][i]);

            for (let n = this.layers.length - 1; n >= 0; n--) {
                A = this.layers[n].getAdjustments(A);
            }
        });
        const maxGradient = 10;
        const gradientLengthSquared = this.layers.reduce((p, layer) => p + layer.dbias.reduce((s, b) => s + b ** 2, 0) + layer.dweights.reduce((s, row) => s + row.reduce((s2, w) => s2 + w ** 2, 0), 0), 0);
        const clipAmount = Math.min(1, maxGradient / Math.sqrt(gradientLengthSquared));
        // const clipAmount = 1;
        for (let n = this.layers.length - 1; n >= 0; n--) {
            this.layers[n].adjust(clipAmount * alpha / inputs.length);
        }
    }
});

const newQuadraticNeuralNet = (layerCounts, randomRange = 1) => {
    const neuralNet = newNeuralNet(layerCounts, randomRange);
    neuralNet.layers = layerCounts.slice(1).map((layerCount, i) => newQuadraticLayer(layerCounts[i], layerCount, randomRange));
    return neuralNet;
};

const newLinearLayer = (inputSize, outputSize, randomRange, sigma = x => x, dsigma = y => 1) => ({
    weights: Array(outputSize).fill().map(() => Array(inputSize).fill().map(() => initRand(randomRange))),
    bias: Array(outputSize).fill().map(() => [initRand(randomRange)]),
    sigma,
    dsigma,
    // lastOutput: undefined,
    pass(input) {
        this.lastInput = [...input];
        this.lastOutput = matadd(matMult(this.weights, input), this.bias).map(row => [this.sigma(row[0])]);
        return this.lastOutput;
    },
    getAdjustments(A) {
        if (!this.dbias) this.dbias = A.map(() => [0]);
        if (!this.dweights) this.dweights = A.map(() => this.lastInput.map(() => 0));

        const adjust = A.map((a, y) => a * this.dsigma(this.lastOutput[y][0]));

        this.dbias = adjust.map((a, y) => [this.dbias[y][0] + a]);
        this.dweights = adjust.map((a, y) => this.lastInput.map((i, x) => this.dweights[y][x] + i[0] * a));
        return matMult([adjust], this.weights)[0];
    },
    adjust(alpha) {
        this.bias = this.bias.map(([bias], i) => [bias - alpha * this.dbias[i][0]]);
        this.weights = this.weights.map((row, i) => row.map((weight, j) => weight - alpha * this.dweights[i][j]));
        this.dbias = undefined;
        this.dweights = undefined;
    }
});

const newQuadraticLayer = (inputSize, outputSize, randomRange) => ({
    a: randomTensor([outputSize, inputSize, inputSize], randomRange),
    b: randomTensor([outputSize, inputSize], randomRange),
    c: randomTensor([outputSize], randomRange),
    pass(input) {
        this.lastInput = [...input];
        this.lastOutput = elementWise([outputSize], ([i]) =>
            sumOverIndices([inputSize, inputSize], ([j, k]) => this.a.get([i, j, k]) * input[j] * input[k]) +
            sumOverIndices([inputSize], ([j]) => this.b.get([i, j]) * input[j]) +
            this.c.get(i)
        ).data;
        return this.lastOutput;
    },
    getAdjustments(A) {
        if (!this.da) this.da = zerosTensor([outputSize, inputSize, inputSize]);
        if (!this.db) this.db = zerosTensor([outputSize, inputSize]);
        if (!this.dc) this.dc = zerosTensor([outputSize]);

        this.da = elementWise([outputSize, inputSize, inputSize], ([i, j, k]) => A[i] * this.lastInput[j] * this.lastInput[k] + this.da.get([i, j, k]));
        this.db = elementWise([outputSize, inputSize], ([i, j]) => A[i] * this.lastInput[j] + this.db.get([i, j]));
        this.dc = elementWise([outputSize], ([i]) => A[i] + this.dc.get(i));
        return elementWise([inputSize], ([j]) => sumOverIndices([outputSize], ([i]) => this.b.get([i, j]) + sumOverIndices([inputSize], ([k]) => (this.a.get([i, j, k]) + this.a.get([i, k, j])) * this.lastInput[k]))).data;
    },
    adjust(alpha) {
        this.a = elementWise([outputSize, inputSize, inputSize], ([i, j, k]) => this.a.get([i, j, k]) - alpha * this.da.get([i, j, k]));
        this.a = elementWise([outputSize, inputSize], ([i, j]) => this.b.get([i, j]) - alpha * this.db.get([i, j]));
        this.a = elementWise([outputSize], ([i]) => this.c.get([i]) - alpha * this.dc.get([i]));
        delete this.da;
        delete this.db;
        delete this.dc;
    },
});

// Incomplete

const newConvolutionalNeuralNet = (kernels, randomRange) => ({
    name: Math.random().toString(36).substring(2),
    kernelLayers: kernels.map(([count, width, height, padding, stride]) => Array(count).fill().map(() => newKernel(width, height, randomRange, padding, stride))),
    kernelCounts: kernels,
    pass(input) { },
    error(input, target) {
        return 0.5 * this.pass(input).reduce((p, c, i) => p + (c - target[i]) ** 2, 0);
    },
    backPropMulti(inputs, targets, alpha) {
        // inputs.forEach((input, j) => {
        //     const target = targets[j];
        //     let A = this.pass(input).map((y, i) => (y - target[i]) * y * (1 - y));
        //     for (let n = this.layers.length - 1; n >= 0; n--) {
        //         A = this.layers[n].getAdjustments(A, alpha);
        //     }
        // });
        // for (let n = this.layers.length - 1; n >= 0; n--) {
        //     this.layers[n].adjust(alpha / inputs.length);
        // }
    }
});

const newKernel = (width, height, range, padding = 0, stride = 1) => ({
    width,
    height,
    padding,
    stride,
    kernel: Array(height).fill().map(() => Array(width).fill().map(() => initRand(range))),
    sigma: ((x) => 1 / (1 + Math.exp(-x))),
    pass(input) {
        const output = [];
        for (let y = -this.padding; y + this.width <= input.length + this.padding; y += stride) {
            output.push([]);
            for (let x = -this.padding; x + this.width <= input[0].length + this.padding; x += stride) {
                let value = 0;
                for (let dy = Math.min(-y, 0); dy < Math.max(input.length - y, this.height); dy++) {
                    for (let dx = Math.min(-x, 0); dx < Math.max(input[0].length - x, this.width); dx++) {
                        value += input[y + dx][x + dx] * kernel[dy][dx];
                    }
                }
                output[y + this.padding].push(this.sigma(value));
            }
        }
        return output;
    }
});

// possible activation functions

const sigmoid = x => 1 / (1 + Math.exp(-x));
const dsigmoid = y => y * (1 - y);

const ReLU = x => x > 0 ? x : 0;
const dReLU = y => y > 0 ? 1 : 0;

const clip = x => Math.max(0, Math.min(1, x));
const dclip = y => y === 1 || y === 0 ? 0 : 1;

const lrelu = (alpha) => x => x > 0 ? x : alpha * x;
const dlrelu = (alpha) => y => y > 0 ? 1 : alpha;