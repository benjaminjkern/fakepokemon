const newNeuralNet = (layers) => ({
    name: Math.random().toString(36).substring(2),
    layers,
    batchNormalize: true,
    maxGradient: 1,
    pass(batch) {
        return this.layers.reduce((currentValues, layer, i) => {
            const results = layer.pass(currentValues);
            if (!this.batchNormalize || results.length === 1 || i === this.layers.length - 1) return results;

            layer.mean = zerosTensor(results[0].dim);
            layer.squareMean = zerosTensor(results[0].dim);
            layer.stdev = zerosTensor(results[0].dim);
            elementWise(results[0].dim, (index) => {
                for (const result of results) {
                    layer.mean.set(index, layer.mean.get(index) + result.get(index) / result.length);
                    layer.squareMean.set(index, layer.mean.get(index) + (result.get(index) ** 2) / result.length);
                }
                layer.stdev.set(index, Math.sqrt(layer.squareMean.get(index) - layer.mean.get(index) ** 2) || 1);

                for (const result of results) {
                    result.set(index, (result.get(index) - layer.mean.get(index)) / layer.stdev.get(index));
                }
            });
            return results;
        }, batch);
    },
    error(batch, targets) {
        const results = this.pass(batch);
        this.lastError = sumOverIndices([results.length], ([b]) =>
            sumOverIndices(results[t].dim, index => (results[b].get(index) - targets[b].get(index)) ** 2)
        ) * 0.5 / batch.length;
        return this.lastError;
    },
    backProp(batch, targets, alpha) {
        const results = this.pass(batch);
        let A = Array(results.length).fill().map((_, b) =>
            elementWise(results[b].dim, (i) => results[b].get(i) - targets[b].get(i))
        );
        for (let n = this.layers.length - 1; n >= 0; n--) {
            A = this.layers[n].getAdjustments(A);
        }

        const effectiveAlpha = alpha * Math.log(this.lastError + 1);

        const gradientLength = Math.sqrt(this.layers.reduce((p, layer) => p + layer.gradientLengthSquared(), 0));

        if (gradientLength === 0) {
            this.layers.forEach(layer => layer.nudge(effectiveAlpha));
            console.log("Network has reached local minimum! Nudging!");
            return;
        }

        const clipAmount = Math.min(1, this.maxGradient / gradientLength);

        for (let n = this.layers.length - 1; n >= 0; n--) {
            this.layers[n].adjust(clipAmount * effectiveAlpha);
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

const newConvolutionalNeuralNet = (inputChannels, inputSize, kernelLayerSpecs, randomRange = 1) => {
    const neuralNet = newNeuralNet([]);
    for (const [i, { channels, kernelSpecs }] of kernelLayerSpecs.entries()) {
        let startChannels;
        if (i === 0) {
            kernelSpecs.inputSize = inputSize;
            startChannels = inputChannels;
        } else {
            kernelSpecs.inputSize = neuralNet.layers[neuralNet.layers.length - 1].kernelSpecs.outputSize;
            startChannels = kernelLayerSpecs[i - 1].channels;
        }
        neuralNet.layers.push(newKernelLayer(startChannels, channels, kernelSpecs, i === kernelLayerSpecs.length - 1 ? linear : ReLU, randomRange));
        console.log(neuralNet.layers[neuralNet.layers.length - 1].kernelSpecs.outputSize, channels, channels * neuralNet.layers[neuralNet.layers.length - 1].kernelSpecs.outputSize[0] * neuralNet.layers[neuralNet.layers.length - 1].kernelSpecs.outputSize[1]);
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

        return elementWise([inputSize], ([j]) => sumOverIndices([outputSize], ([i]) => adjust.get(i) * this.b.get([i, j])));
    },
    adjust(alpha) {
        this.b = elementWise([outputSize, inputSize], ([i, j]) => this.b.get([i, j]) - alpha * this.db.get([i, j]));
        this.a = elementWise([outputSize], ([i]) => this.a.get([i]) - alpha * this.da.get([i]));
        delete this.db;
        delete this.da;
    },
    gradientLengthSquared() {
        return sumOverIndices([outputSize], ([i]) => this.da.get(i) ** 2) + sumOverIndices([outputSize, inputSize], ([i, j]) => this.db.get([i, j]) ** 2);
    },
    nudge(mutation) {
        this.b = elementWise([outputSize, inputSize], ([i, j]) => this.b.get([i, j]) + mutation * (Math.random() * 2 - 1));
        this.a = elementWise([outputSize], ([i]) => this.a.get([i]) + mutation * (Math.random() * 2 - 1));
    }
});

const newQuadraticLayer = (inputSize, outputSize, randomRange = 1) => ({
    c: randomTensor([outputSize, inputSize, inputSize], randomRange),
    b: randomTensor([outputSize, inputSize], randomRange),
    a: randomTensor([outputSize], randomRange),
    pass(input) {
        this.lastInput = input;
        this.lastOutput = elementWise([outputSize], ([i]) =>
            sumOverIndices([inputSize, inputSize], ([j, k]) => this.c.get([i, j, k]) * input.get(j) * input.get(k)) +
            sumOverIndices([inputSize], ([j]) => this.b.get([i, j]) * input.get(j)) +
            this.a.get(i)
        );
        return this.lastOutput;
    },
    getAdjustments(A) {
        if (!this.dc) this.dc = zerosTensor([outputSize, inputSize, inputSize]);
        if (!this.db) this.db = zerosTensor([outputSize, inputSize]);
        if (!this.da) this.da = zerosTensor([outputSize]);

        this.dc = elementWise([outputSize, inputSize, inputSize], ([i, j, k]) => A.get(i) * this.lastInput.get(j) * this.lastInput.get(k) + this.dc.get([i, j, k]));
        this.db = elementWise([outputSize, inputSize], ([i, j]) => A.get(i) * this.lastInput.get(j) + this.db.get([i, j]));
        this.da = elementWise([outputSize], ([i]) => A[i] + this.da.get(i));
        return elementWise([inputSize], ([j]) => sumOverIndices([outputSize], ([i]) => A.get(i) * (this.b.get([i, j]) + sumOverIndices([inputSize], ([k]) => (this.c.get([i, j, k]) + this.c.get([i, k, j])) * this.lastInput.get(k))))).data;
    },
    adjust(alpha) {
        this.c = elementWise([outputSize, inputSize, inputSize], ([i, j, k]) => this.c.get([i, j, k]) - alpha * this.dc.get([i, j, k]));
        this.b = elementWise([outputSize, inputSize], ([i, j]) => this.b.get([i, j]) - alpha * this.db.get([i, j]));
        this.a = elementWise([outputSize], ([i]) => this.a.get([i]) - alpha * this.da.get([i]));
        delete this.dc;
        delete this.db;
        delete this.da;
    },
    gradientLengthSquared() {
        // TODO: Implement
        return 1;
    }
});

const newKernelLayer = (inputChannels, outputChannels, kernelSpecs, sigma, randomRange = 1) => {
    const processedKernelSpecs = {
        ...kernelSpecs,
    };
    default2d(processedKernelSpecs, 'padding', 0);
    default2d(processedKernelSpecs, 'innerPadding', 0);
    default2d(processedKernelSpecs, 'stride', 1);
    processedKernelSpecs.trueInputSize = [0, 1].map(i => 2 * processedKernelSpecs.padding[i] + processedKernelSpecs.inputSize[i] + processedKernelSpecs.innerPadding[i] * (processedKernelSpecs.inputSize[i] - 1));
    processedKernelSpecs.outputSize = [0, 1].map(i => 1 + Math.floor((processedKernelSpecs.trueInputSize[i] - processedKernelSpecs.kernelSize[i]) / processedKernelSpecs.stride[i]));

    return {
        inputChannels,
        kernelSpecs: processedKernelSpecs,
        kernels: Array(outputChannels).fill().map(() => Array(inputChannels).fill().map(() => newKernel(processedKernelSpecs, randomRange))),
        bias: Array(outputChannels).fill().map(() => initRand(randomRange)),
        sigma,
        pass(inputTensor) {
            const processedInputs = Array(inputChannels).fill().map((_, channel) => {
                return elementWise(inputTensor.dim.slice(0, 2), ([i, j]) => inputTensor.get([i, j, channel]));
            });

            // Currently this is the slowest bit, dont really know a way to make it faster lol
            const outputtedKernelValues = this.kernels.map(outputs => outputs.map((kernel, inputChannel) => kernel.pass(processedInputs[inputChannel])));


            this.lastOutput = elementWise([...processedKernelSpecs.outputSize, outputChannels], ([ox, oy, i]) =>
                this.sigma(
                    this.bias[i] +
                    sumOverIndices([inputChannels], ([j]) => outputtedKernelValues[i][j].get([ox, oy]))
                )
            );
            return this.lastOutput;
        },
        getAdjustments(A) {
            if (!this.dbias) this.dbias = zerosTensor([outputChannels]);
            const adjust = elementWise([...processedKernelSpecs.outputSize, outputChannels], ([ox, oy, i]) => A.get([ox, oy, i]) * this.sigma(this.lastOutput.get([ox, oy, i]), true));
            this.dbias = elementWise([outputChannels], ([i]) => sumOverIndices(processedKernelSpecs.outputSize, ([ox, oy]) => this.dbias.get(i) + adjust.get([ox, oy, i])));
            for (let n = 0; n < outputChannels; n++) {
                for (let m = 0; m < inputChannels; m++) {
                    this.kernels[n][m].getAdjustments(adjust, n);
                }
            }

            const newA = zerosTensor([...processedKernelSpecs.inputSize, inputChannels]);
            elementWise([inputChannels], ([j]) => this.kernels[0][0].inputOutputLoop(([ox, oy, r, s], kernelIndex) => {
                newA.set([r, s, j],
                    newA.get([r, s, j]) +
                    sumOverIndices([outputChannels], ([i]) => adjust.get([ox, oy, i]) * this.kernels[i][j].adjustment.get([ox, oy, r, s]))
                )
            }));
            return newA;
        },
        adjust(alpha) {
            for (let n = 0; n < outputChannels; n++) {
                this.bias[n] -= this.dbias.get(n) * alpha;
                for (let m = 0; m < inputChannels; m++) {
                    this.kernels[n][m].adjust(alpha);
                }
            }
            delete this.dbias;
        },
        gradientLengthSquared() {
            return sumOverIndices([outputChannels], ([n]) => this.dbias.get(n) ** 2) + sumOverIndices([outputChannels, inputChannels], ([n, m]) => this.kernels[n][m].gradientLengthSquared());
        },
        nudge(mutation) {
            for (let n = 0; n < outputChannels; n++) {
                this.bias[n] += mutation * (Math.random() * 2 - 1);
                for (let m = 0; m < inputChannels; m++) {
                    this.kernels[n][m].nudge(mutation);
                }
            }
        }
    }
};

const default2d = (kernelSpecs, attribute, defaultValue) => {
    if (kernelSpecs[attribute] === undefined) kernelSpecs[attribute] = [defaultValue, defaultValue];
    if (typeof kernelSpecs[attribute] === 'number') kernelSpecs[attribute] = [kernelSpecs[attribute], kernelSpecs[attribute]];
}

const newPoolingLayer = (inputChannels, outputChannels, poolFunc) => { };

const newKernel = (kernelSpecs, randomRange) => {
    return {
        ...kernelSpecs,

        kernel: randomTensor(kernelSpecs.kernelSize, randomRange),
        pass(inputTensor) {
            this.lastInput = inputTensor;
            this.lastOutput = zerosTensor(this.outputSize);
            this.inputOutputLoop(([ox, oy, r, s], kernelIndex) => {
                this.lastOutput.set([ox, oy], this.lastOutput.get([ox, oy]) +
                    this.kernel.get(kernelIndex) *
                    this.lastInput.get([r, s]));
            });
            return this.lastOutput;
        },
        getAdjustments(A, i) {
            if (!this.dkernel) {
                this.dkernel = zerosTensor(this.kernelSize);
            }

            this.adjustment = zerosTensor([...this.outputSize, ...this.inputSize]);
            this.inputOutputLoop(([ox, oy, r, s], kernelIndex) => {
                this.dkernel.set(kernelIndex, this.dkernel.get(kernelIndex) + A.get([ox, oy, i]) * this.lastInput.get([r, s]));
                this.adjustment.set([ox, oy, r, s], this.kernel.get(kernelIndex));
            });
        },
        inputOutputLoop(func) {
            for (let ox = 0; ox < this.outputSize[0]; ox++) {
                for (let oy = 0; oy < this.outputSize[1]; oy++) {
                    const rMin = Math.max(0, Math.ceil((ox * this.stride[0] - this.padding[0]) / (this.innerPadding[0] + 1)));
                    const rMax = Math.min(this.inputSize[0], Math.ceil((this.kernelSize[0] + ox * this.stride[0] - this.padding[0]) / (this.innerPadding[0] + 1)));
                    const sMin = Math.max(0, Math.ceil((oy * this.stride[1] - this.padding[1]) / (this.innerPadding[1] + 1)));
                    const sMax = Math.min(this.inputSize[1], Math.ceil((this.kernelSize[1] + oy * this.stride[1] - this.padding[1]) / (this.innerPadding[1] + 1)));
                    for (let r = rMin; r < rMax; r++) {
                        for (let s = sMin; s < sMax; s++) {
                            const kernelIndex = [this.padding[0] + (this.innerPadding[0] + 1) * r - ox * this.stride[0], this.padding[1] + (this.innerPadding[1] + 1) * s - oy * this.stride[1]];
                            func([ox, oy, r, s], kernelIndex);
                        }
                    }
                }
            }
        },
        adjust(alpha) {
            this.kernel = elementWise(this.kernelSize, ([x, y]) => this.kernel.get([x, y]) - alpha * this.dkernel.get([x, y]));
            delete this.dkernel;
        },
        gradientLengthSquared() {
            return sumOverIndices(this.kernelSize, ([x, y]) => this.dkernel.get([x, y]) ** 2);
        },
        nudge(mutation) {
            elementWise(this.kernelSize, ([x, y]) => {
                this.kernel.set([x, y], this.kernel.get([x, y]) + mutation * (Math.random() * 2 - 1));
            });
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

// Todo: Make work
// const dropout = (func, rate) =>  (x, derivative) => {
//     if (Math.random() < rate) return 0;
//     return func(x, derivative)
// }

// This is just for convenience, if someone wants to create their own activation function all they need to do is provide a forward function and an activation
const createActivationFunc = (reg, deriv) => (x, derivative = false) => {
    if (!derivative) return reg(x);
    return deriv(x);
}