const newNeuralNet = (layers) => ({
    name: Math.random().toString(36).substring(2),
    layers,
    pass(input) {
        if (!input.dim) return this.pass(newTensor([input.length], input));
        return this.layers.reduce((p, layer) => layer.pass(p), input);
    },
    error(inputs, targets) {
        this.lastError = inputs.reduce((acc, input, t) => {
            const target = targets[t];
            return acc + this.pass(input).data.reduce((p, c, i) => p + (c - target[i]) ** 2, 0);
        }, 0) * 0.5 / inputs.length;
        return this.lastError;
    },
    backPropMulti(inputs, targets, alpha) {
        inputs.forEach((input, j) => {
            // TODO: derror
            let A = this.pass(input).map((y, i) => y - targets[j][i]);

            for (let n = this.layers.length - 1; n >= 0; n--) {
                A = this.layers[n].getAdjustments(A);
            }
        });
        // this.layers.forEach(layer => layer.clipGradient(10));
        for (let n = this.layers.length - 1; n >= 0; n--) {
            this.layers[n].adjust(alpha / inputs.length * Math.log(this.lastError + 1));
        }
    }
});

const mutateLinearNeuralNet = (oldNeuralNet, mutation = 1) => {
    const newNet = newLinearNeuralNet(oldNeuralNet.layerCounts, mutation);
    newNet.layers.forEach((layer, i) => {
        layer.weights = matadd(layer.weights, oldNeuralNet.layers[i].weights);
        layer.bias = matadd(layer.bias, oldNeuralNet.layers[i].bias);
    })
    return newNet;
}

// Different types of neural nets
const newLinearNeuralNet = (layerCounts, randomRange = 1, activation = ReLU, finalActivation = activation) => {
    return newNeuralNet(layerCounts.slice(1).map((layerCount, i) => newLinearLayer(layerCounts[i], layerCount, randomRange, i === layerCounts.length - 2 ? finalActivation : activation)));
}

const newQuadraticNeuralNet = (layerCounts, randomRange = 1) => {
    return newNeuralNet(layerCounts.slice(1).map((layerCount, i) => newQuadraticLayer(layerCounts[i], layerCount, randomRange)));
};

const newConvolutionalNeuralNet = (kernelLayerSpecs, randomRange = 1) => {
    const neuralNet = newNeuralNet([]);
    for (const [i, { channels, kernelSpecs }] of kernelLayerSpecs.slice(0, kernelLayerSpecs.length - 1).entries()) {
        this.layers.push(newKernelLayer(channels, kernelLayerSpecs[i + 1], kernelSpecs, ReLU, randomRange))
    }
    return neuralNet;
}

// Different types of layers
const newLinearLayer = (inputSize, outputSize, randomRange = 1, sigma = linear) => ({
    b: randomTensor([outputSize, inputSize], randomRange),
    a: randomTensor([outputSize], randomRange),
    sigma,
    pass(input) {
        this.lastInput = input;
        this.lastOutput = elementWise([outputSize], ([i]) =>
            sumOverIndices([inputSize], ([j]) => this.b.get([i, j]) * input[j]) +
            this.a.get(i)
        ).map(this.sigma);
        return this.lastOutput;
    },
    getAdjustments(A) {
        if (!this.db) this.db = zerosTensor([outputSize, inputSize]);
        if (!this.da) this.da = zerosTensor([outputSize]);

        const adjust = elementWise([outputSize], ([i]) => A.get(i) * this.sigma(this.lastOutput.get(i), true));

        this.db = elementWise([outputSize, inputSize], ([i, j]) => adjust.get(i) * this.lastInput.get(j) + this.db.get([i, j]));
        this.da = elementWise([outputSize], ([i]) => adjust.get(i) + this.da.get(i));

        return elementWise([inputSize], ([j]) => sumOverIndices([outputSize], ([i]) => adjust.get(i) * this.b.get([i, j]))).data;
    },
    adjust(alpha) {
        this.b = elementWise([outputSize, inputSize], ([i, j]) => this.b.get([i, j]) - alpha * this.db.get([i, j]));
        this.a = elementWise([outputSize], ([i]) => this.a.get([i]) - alpha * this.da.get([i]));
        delete this.db;
        delete this.da;
    },
    // clipGradient(maxGradient) {
    //     const gradientLengthSquared = sumOverIndices([outputSize], ([i]) => this.da.get(i) ** 2) + sumOverIndices([outputSize, inputSize], ([i, j]) => this.db.get([i, j]) ** 2);
    //     const clipAmount = Math.min(1, maxGradient / Math.sqrt(gradientLengthSquared));
    //     if (clipAmount < 1) {
    //         this.db = elementWise([outputSize, inputSize], ([i, j]) => this.db.get([i, j]) * clipAmount);
    //         this.da = elementWise([outputSize], ([i]) => this.da.get(i) * clipAmount);
    //     }
    // }
});

const newQuadraticLayer = (inputSize, outputSize, randomRange = 1) => ({
    c: randomTensor([outputSize, inputSize, inputSize], randomRange),
    b: randomTensor([outputSize, inputSize], randomRange),
    a: randomTensor([outputSize], randomRange),
    pass(input) {
        this.lastInput = [...input];
        this.lastOutput = elementWise([outputSize], ([i]) =>
            sumOverIndices([inputSize, inputSize], ([j, k]) => this.c.get([i, j, k]) * input[j] * input[k]) +
            sumOverIndices([inputSize], ([j]) => this.b.get([i, j]) * input[j]) +
            this.a.get(i)
        ).data;
        return this.lastOutput;
    },
    getAdjustments(A) {
        if (!this.dc) this.dc = zerosTensor([outputSize, inputSize, inputSize]);
        if (!this.db) this.db = zerosTensor([outputSize, inputSize]);
        if (!this.da) this.da = zerosTensor([outputSize]);

        this.dc = elementWise([outputSize, inputSize, inputSize], ([i, j, k]) => A[i] * this.lastInput[j] * this.lastInput[k] + this.dc.get([i, j, k]));
        this.db = elementWise([outputSize, inputSize], ([i, j]) => A[i] * this.lastInput[j] + this.db.get([i, j]));
        this.da = elementWise([outputSize], ([i]) => A[i] + this.da.get(i));
        return elementWise([inputSize], ([j]) => sumOverIndices([outputSize], ([i]) => A[i] * (this.b.get([i, j]) + sumOverIndices([inputSize], ([k]) => (this.c.get([i, j, k]) + this.c.get([i, k, j])) * this.lastInput[k])))).data;
    },
    adjust(alpha) {
        this.c = elementWise([outputSize, inputSize, inputSize], ([i, j, k]) => this.c.get([i, j, k]) - alpha * this.dc.get([i, j, k]));
        this.b = elementWise([outputSize, inputSize], ([i, j]) => this.b.get([i, j]) - alpha * this.db.get([i, j]));
        this.a = elementWise([outputSize], ([i]) => this.a.get([i]) - alpha * this.da.get([i]));
        delete this.dc;
        delete this.db;
        delete this.da;
    },
});

const newKernelLayer = (inputChannels, outputChannels, kernelSpecs, sigma, randomRange = 1) => {
    const processedKernelSpecs = {
        ...kernelSpecs,
        padding: typeof kernelSpecs.padding === 'object' ? kernelSpecs.padding : [kernelSpecs.padding, kernelSpecs.padding],
        innerPadding: typeof kernelSpecs.innerPadding === 'object' ? kernelSpecs.innerPadding : [kernelSpecs.innerPadding, kernelSpecs.innerPadding],
        stride: typeof kernelSpecs.stride === 'object' ? kernelSpecs.stride : [kernelSpecs.stride, kernelSpecs.stride],
        trueInputSize: [0, 1].map(i => 2 * this.padding[i] + this.inputSize[i] + this.innerPadding[i] * (this.inputSize[i] - 1)),
        outputSize: [0, 1].map(i => 1 + Math.floor((this.trueInputSize[i] - this.kernelSize[i]) / this.stride[i])),
    };
    processedKernelSpecs.LField = zerosTensor([...processedKernelSpecs.trueInputSize, 2]);
    for (let r = 0; r < processedKernelSpecs.inputSize[0]; r++) {
        for (let s = 0; s < processedKernelSpecs.inputSize[0]; s++) {
            processedKernelSpecs.LField.set([processedKernelSpecs.padding[0] + x * processedKernelSpecs.innerPadding[0], processedKernelSpecs.padding[1] + y * processedKernelSpecs.innerPadding[1], 0], r);
            processedKernelSpecs.LField.set([processedKernelSpecs.padding[0] + x * processedKernelSpecs.innerPadding[0], processedKernelSpecs.padding[1] + y * processedKernelSpecs.innerPadding[1], 1], s);
        }
    }

    return {
        kernels: Array(outputChannels).fill().map(() => Array(inputChannels).fill().map(() => newKernel(processedKernelSpecs, randomRange))),
        bias: Array(outputChannels).fill().map(() => initRand(randomRange)),
        sigma,
        pass(inputTensor) {
            if (inputTensor.dim === undefined) inputTensor = newTensor([1, 1, inputTensor.length], inputTensor);
            // needs to return (outputWidth x outputHeight) x outputChannels
            const outputtedKernelValues = this.kernels.map(outputs => outputs.map(kernel => kernel.pass(inputTensor)));
            this.lastOutput = elementWise([...processedKernelSpecs.outputSize, outputChannels], ([i, ox, oy]) =>
                this.sigma(
                    this.bias[i] +
                    sumOverIndices([inputChannels], ([j]) => outputtedKernelValues[i][j].get([ox, oy]))
                )
            );
            return this.lastOutput;
        },
        getAdjustments(A) {
            if (!this.dbias) zerosTensor([outputChannels]);
            const adjust = elementWise([...processedKernelSpecs.outputSize, outputChannels], ([i, ox, oy]) => A.get([i, ox, oy]) * this.sigma(this.lastOutput.get([i, ox, oy]), true));
            this.dbias = elementWise([outputChannels], ([i]) => sumOverIndices(processedKernelSpecs.outputSize, ([ox, oy]) => this.dbias.get(i) + adjust.get([i, ox, oy])));
            for (let n = 0; n < outputChannels; n++) {
                for (let m = 0; m < inputChannels; m++) {
                    this.kernels[n][m].getAdjustments(adjust, n);
                }
            }

            return elementWise([inputChannels, ...processedKernelSpecs.inputSize], ([j, r, s]) => sumOverIndices([outputChannels], ([i, ox, oy]) => adjust.get([i, ox, oy]) * this.kernels[i][j].adjustment.get([ox, oy, r, s])));
        },
        adjust(alpha) {
            for (let n = 0; n < outputChannels; n++) {
                this.bias[n] -= this.dbias.get(n) * alpha;
                for (let m = 0; m < inputChannels; m++) {
                    this.kernels[n][m].adjust(alpha);
                }
            }
            delete this.dbias;
        }
    }
};

const newPoolingLayer = (inputChannels, outputChannels, poolFunc) => {};

const newKernel = (kernelSpecs, randomRange) => {
    return {
        ...kernelSpecs,

        kernel: randomTensor(this.kernelSize, randomRange),
        pass(inputTensor) {
            // Add inner and outer padding
            this.lastProcessedInput = zerosTensor(this.trueInputSize);
            for (let x = 0; x < inputTensor.dim[0]; x++) {
                for (let y = 0; y < inputTensor.dim[1]; y++) {
                    this.lastProcessedInput.set([this.padding[0] + x * this.innerPadding[0], this.padding[1] + y * this.innerPadding[1]], inputTensor.get([x, y]));
                }
            }
            return elementWise(this.outputSize, ([ox, oy]) =>
                sumOverIndices(this.kernelSize, ([x, y]) =>
                    this.kernel.get([x, y]) * this.lastProcessedInput.get([x + this.stride[0] * ox, y + this.stride[1] * oy])
                )
            );
        },
        getAdjustments(A, i) {
            this.dkernel = elementWise(this.outputSize, ([ox, oy]) =>
                sumOverIndices(this.kernelSize, ([x, y]) =>
                    A.get([i, ox, oy]) * this.lastProcessedInput.get([x + this.stride[0] * ox, y + this.stride[1] * oy])
                )
            );
            this.adjustment = elementWise([...this.outputSize, ...this.inputSize], ([ox, oy, r, s]) => sumOverIndices(this.kernelSize, ([x, y]) =>
                this.kernel.get([x, y]) * this.L(ox, oy, x, y, r, s)
            ));
        },
        adjust(alpha) {
            this.kernel = elementWise(this.kernelSize, ([x, y]) => this.kernel.get([x, y]) - alpha * this.dkernel.get([x, y]));
            delete this.dkernel;
        },
        L(ox, oy, x, y, r, s) {
            return kernelSpecs.LField.get([x + this.stride[0] * ox, y + this.stride[1] * oy, 0]) === r && LField.get([x + this.stride[0] * ox, y + this.stride[1] * oy, 1]) === s;
        }
    }
};

// possible activation functions, when derivative is set to true, it is expecting the output rather than the input as x.
// This is just because for many of the functions it is faster to compute the derivative from the output than it is from the input.

const sigmoid = (x, derivative) => {
    if (!derivative) return 1 / (1 + Math.exp(-x));
    return x * (1 - x);
}
const ReLU = (x, derivative) => {
    if (!derivative) return x > 0 ? x : 0;
    return x > 0 ? 1 : 0
}
const clip = (x, derivative) => {
    if (!derivative) return Math.max(0, Math.min(1, x));
    return x === 1 || x === 0 ? 0 : 1;
}
const lrelu = (alpha) => (x, derivative) => {
    if (!derivative) return x > 0 ? x : alpha * x;
    if (alpha < 0) throw "Leaky Relu cannot determine the derivative from the output when alpha is less than 0!";
    return x > 0 ? 1 : alpha;
}
const tanh = (x, derivative) => {
    if (!derivative) {
        if (x >= 19) return 1;
        if (x <= -19) return -1;
        const ep = Math.exp(2 * x);
        return (ep - 1) / (ep + 1);
    }
    return 1 - x ** 2;
}
const linear = (x, derivative = false) => {
    if (!derivative) return x;
    return 1;
}

// This is just for convenience, if someone wants to create their own activation function all they need to do is provide a forward function and an activation
const createActivationFunc = (reg, deriv) => (x, derivative = false) => {
    if (!derivative) return reg(x);
    return deriv(x);
}