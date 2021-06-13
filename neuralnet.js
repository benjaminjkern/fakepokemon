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
    layers: layerCounts.slice(1).map((layerCount, i) => newLinearLayer(layerCounts[i], layerCount, randomRange)),
    layerCounts,
    pass(input) {
        return this.layers.reduce((p, layer) => layer.pass(p), input.map(i => [i])).flat();
    },
    error(input, target) {
        return 0.5 * this.pass(input).reduce((p, c, i) => p + (c - target[i]) ** 2, 0);
    },
    backPropMulti(inputs, targets, alpha) {
        inputs.forEach((input, j) => {
            const target = targets[j];
            let A = this.pass(input).map((y, i) => (y - target[i]) * y * (1 - y));
            for (let n = this.layers.length - 1; n >= 0; n--) {
                A = this.layers[n].getAdjustments(A, alpha);
            }
        });
        for (let n = this.layers.length - 1; n >= 0; n--) {
            this.layers[n].adjust(alpha / inputs.length);
        }
    }
});

const newLinearLayer = (inputSize, outputSize, randomRange) => ({
    weights: Array(outputSize).fill().map(() => Array(inputSize).fill().map(() => initRand(randomRange))),
    bias: Array(outputSize).fill().map(() => [initRand(randomRange)]),
    sigma: ((x) => 1 / (1 + Math.exp(-x))),
    lastOutput: undefined,
    pass(input) {
        this.lastInput = [...input];
        this.lastOutput = matadd(matMult(this.weights, input), this.bias).map(row => [this.sigma(row[0])]);
        return this.lastOutput;
    },
    getAdjustments(A) {
        this.dbias = A.map((a, y) => [((this.dbias ? this.dbias[y] : 0) - 0) + (a - 0)]);
        this.dweights = A.map((a, y) => this.lastInput.map((i, x) => ((this.dweights ? this.dweights[y][x] : 0) - 0) + (i - 0) * (a - 0)));
        return matMult([A], this.weights)[0].map((y, i) => y * this.lastInput[i] * (1 - this.lastInput[i]));
    },
    adjust(alpha) {
        this.bias = this.bias.map(([bias], i) => [bias - alpha * this.dbias[i]]);
        this.weights = this.weights.map((row, i) => row.map((weight, j) => weight - alpha * this.dweights[i][j]));
        this.dbias = undefined;
        this.dweights = undefined;
    }
});

const newConvolutionalNeuralNet = (kernels, randomRange) => ({
    name: Math.random().toString(36).substring(2),
    kernelLayers: kernels.map(([count, width, height, padding, stride]) => Array(count).fill().map(() => newKernel(width, height, randomRange, padding, stride))),
    kernelCounts: kernels,
    pass(input) {},
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