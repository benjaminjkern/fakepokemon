class NeuralNet {
    constructor(layers, name = Math.random().toString(36).substring(2), options = {}) {
        this.layers = layers;
        this.name = name;

        this.nudgeOnFlat = options.nudgeOnFlat || true;
        this.maxGradient = options.maxGradient || 1;
    }
    pass(batch) {
        return this.layers.reduce((currentValues, layer, i) => {
            return layer.pass(currentValues);
        }, batch);
    }
    error(batch, targets) {
        const results = this.pass(batch);
        this.lastError = sumOverIndices([results.length], ([b]) =>
            sumOverIndices(results[t].dim, index => (results[b].get(index) - targets[b].get(index)) ** 2)
        ) * 0.5 / batch.length;
        return this.lastError;
    }
    backProp(batch, targets, alpha) {
        const results = this.pass(batch);
        let A = Array(results.length).fill().map((_, b) =>
            elementWise(results[b].dim, (i) => results[b].get(i) - targets[b].get(i))
        );
        for (let n = this.layers.length - 1; n >= 0; n--) {
            A = this.layers[n].setAdjustments(A);
        }

        const effectiveAlpha = alpha * Math.log(this.lastError + 1) / batch.length;

        const gradientLength = Math.sqrt(this.layers.reduce((p, layer) => p + layer.gradientLengthSquared(), 0));

        if (gradientLength === 0) {
            if (this.nudge) {
                console.log("Network has reached local minimum! Nudging!");
                this.layers.forEach(layer => layer.nudge(effectiveAlpha));
            }
            return;
        }

        const clipAmount = Math.min(1, this.maxGradient / gradientLength);

        for (let n = this.layers.length - 1; n >= 0; n--) {
            this.layers[n].adjust(clipAmount * effectiveAlpha);
        }
    }
}


//-------------------------------------------------------------------------------------------------------------------
// Different types of out-of-the-box neural nets
//-------------------------------------------------------------------------------------------------------------------
class LinearNeuralNet extends NeuralNet {
    constructor(layerCounts, randomRange = 1, activation = ReLU, finalActivation = activation) {
        super(layerCounts.slice(1).reduce((p, layerCount, i) => [...p,
        new LinearLayer(layerCounts[i], layerCount, randomRange),
        new BatchNormalizationLayer(layerCount),
        new ActivationLayer(layerCount, i < layerCounts.length - 1 ? activation : finalActivation)
        ], []));
    }
}

class QuadraticNeuralNet extends NeuralNet {
    constructor(layerCounts, randomRange = 1) {
        // You dont really need activations for quadratic neural nets, although you can definitely still have them.
        // These out-of-the-box neural nets dont have them by default though.
        super(layerCounts.slice(1).reduce((p, layerCount, i) => [...p,
        new QuadraticLayer(layerCounts[i], layerCount, randomRange),
        new BatchNormalizationLayer(layerCount)
        ], []));
    }
}

class ConvolutionalNeuralNet extends NeuralNet {
    constructor(inputChannels, inputSize, kernelLayerSpecs, activation = ReLU, finalActivation = activation, randomRange = 1) {
        super([]);
        for (const [i, { channels, kernelSpecs }] of kernelLayerSpecs.entries()) {
            let size = i === 0 ? [...inputSize, inputChannels] : this.layers[this.layers.length - 3].outputSize;
            const layer = new KernelLayer(...size, channels, kernelSpecs);
            console.log(layer.outputSize, dimSize(layer.outputSize));
            this.layers.push(layer);
            this.layers.push(new BatchNormalizationLayer(layer.outputSize));
            this.layers.push(new ActivationLayer(layer.outputSize, i < kernelLayerSpecs.length - 1 ? activation : finalActivation));
        }
    }
}

//-------------------------------------------------------------------------------------------------------------------
// Different types of layers
//-------------------------------------------------------------------------------------------------------------------

class Layer {
    constructor(inputSize, outputSize) {
        if (this.constructor == Layer)
            throw "Abstract Class!";

        if (typeof inputSize === 'number') {
            this.inputSize = [inputSize];
            this.outputSize = [outputSize];
        } else {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
        }
    }

    // Pass a batch of inputs through the layer
    pass(batch) {
        this.lastInput = batch;
        this.lastOutput = batch.map(this.passOne);
        return this.lastOutput;
    }
    // Given the deltas of the layer above it, calculate the derivatives of all fields
    setDerivatives(batchDelta) {
        this.initDerivatives();
        return batchDelta.map(this.setOneDerivative);
    }

    // Abstract methods

    // Pass one input through the layer
    passOne(input) {
        throw "Abstract Method!";
    }
    // Initialize derivatives for backpropagation
    initDerivatives(delta) {
        throw "Abstract Method!";
    }
    // Given the delta of the layer above it, calculate the derivative contribution of one input
    setOneDerivative(delta, batchIdx) {
        throw "Abstract Method!";
    }
    // Adjust all fields according to their calculated derivatives
    adjust(alpha) {
        throw "Abstract Method!";
    }
    // Calculate the length of the gradient of this layer, for use in gradient clipping (Or analysis)
    gradientLengthSquared() {
        throw "Abstract Method!";
    }
    // Randomly adjust all fields
    nudge(alpha) {
        throw "Abstract Method!";
    }
}

class ActivationLayer extends Layer {
    constructor(inputSize, activationFunc) {
        super(inputSize, inputSize);
        this.activationFunc = activationFunc;
    }
    passOne(input) {
        return input.map(this.activationFunc)
    }
    initDerivatives() {
        // There are no fields to adjust in an activation layer
    }
    setOneDerivative(delta, batchIdx) {
        return delta.map((index) => delta.get(index) * this.activationFunc(this.lastOutput[batchIdx].get(index), true));
    }
    adjust(alpha) {
        // There are no fields to adjust in an activation layer
    }
    gradientLengthSquared() {
        // There is no gradient in an activation layer
    }
    nudge(alpha) {
        // There are no fields to nudge in an activation layer
    }
}

class BatchNormalizationLayer extends Layer {
    constructor(inputSize) {
        super(inputSize, inputSize);
    }
    // this one needs to overwrite the pass within layer because it needs access to the batch itself
    pass(batch) {
        if (batch.length === 1) return batch;

        this.mean = zerosTensor(batch[0].dim);
        this.squareMean = zerosTensor(batch[0].dim);
        this.stdev = zerosTensor(batch[0].dim);
        const resultBatch = batch.map(zerosTensor(batch[0].dim));
        elementWise(batch[0].dim, (index) => {
            for (const input of batch) {
                layer.mean.add_to(index, input.get(index) / batch.length);
                layer.squareMean.add_to(index, (input.get(index) ** 2) / result.length);
            }
            layer.stdev.set(index, Math.sqrt(layer.squareMean.get(index) - layer.mean.get(index) ** 2) || 1);

            for (const result of resultBatch) {
                result.set(index, (result.get(index) - this.mean.get(index)) / this.stdev.get(index));
            }
        });
        return resultBatch;
    }

    passOne(input) {
        throw "Must pass entire batch for a batch normalization layer!";
    }
    initDerivatives() {
        // There are no fields to adjust in a batch normalization layer
    }
    setOneDerivative(delta, batchIdx) {
        if (!this.stdev) return delta;
        return delta.map((index) => delta.get(index) / this.stdev.get(index));
    }
    adjust(alpha) {
        // There are no fields to adjust in a batch normalization layer
    }
    gradientLengthSquared() {
        // There is no gradient in a batch normalization layer
    }
    nudge(alpha) {
        // There are no fields to nudge in a batch normalization layer
    }
}

class LinearLayer extends PolyLayer {
    constructor(inputSize, outputSize, randomRange = 1) {
        super(inputSize, outputSize, 1, randomRange);
    }
    passOne(input) {
        return elementWise([outputSize], ([i]) =>
            sumOverIndices([inputSize], ([j]) => this.fields[1].get([i, j]) * input.get(j)) +
            this.fields[0].get(i)
        );
    }
    setOneDerivative(delta, batchIdx) {
        elementWise([outputSize], ([i]) => {
            elementWise([inputSize], ([j]) => {
                this.dfields[1].add_to([i, j], delta.get(i) * this.lastInput[batchIdx].get(j));
            });
            this.dfields[0].add_to(i, delta.get(i));
        });

        return elementWise([inputSize], ([j]) => sumOverIndices([outputSize], ([i]) => delta.get(i) * this.dfields[1].get([i, j])));
    }
}

class QuadraticLayer extends PolyLayer {
    constructor(inputSize, outputSize, randomRange = 1) {
        super(inputSize, outputSize, 2, randomRange);
    }
    passOne(input) {
        return elementWise([outputSize], ([i]) =>
            sumOverIndices([inputSize, inputSize], ([j, k]) => this.fields[2].get([i, j, k]) * input.get(j) * input.get(k)) +
            sumOverIndices([inputSize], ([j]) => this.fields[1].get([i, j]) * input.get(j)) +
            this.fields[0].get(i)
        );
    }
    setOneDerivative(delta, batchIdx) {
        elementWise([outputSize], ([i]) => {
            elementWise([inputSize], ([j]) => {
                elementWise([inputSize], ([k]) => {
                    this.dfields[2].add_to([i, j, k], delta.get(i) * this.lastInput[batchIdx].get(j) * this.lastInput[batchIdx].get(k));
                });
                this.dfields[1].add_to([i, j], delta.get(i) * this.lastInput[batchIdx].get(j));
            });
            this.dfields[0].add_to(i, delta.get(i));
        });

        return elementWise([inputSize], ([j]) => sumOverIndices([outputSize], ([i]) => delta.get(i) * (this.fields[1].get([i, j]) + sumOverIndices([inputSize], ([k]) => (this.fields[2].get([i, j, k]) + this.fields[2].get([i, k, j])) * this.lastInput.get(k)))));
    }
}

class PolyLayer extends Layer {
    constructor(inputSize, outputSize, maxPower, randomRange = 1) {
        super(inputSize, outputSize);
        this.fields = Array(maxPower + 1).fill().map((_, p) => randomTensor([outputSize, ...Array(p).fill(inputSize)], randomRange));
    }
    passOne(input) {
        throw "Not implemented for arbitrary power polynomial layers.";
    }
    initDerivatives() {
        this.dfields = this.fields.map((_, p) => zerosTensor([outputSize, ...Array(p).fill(inputSize)]));
    }
    setOneDerivative(delta, batchIdx) {
        throw "Not implemented for arbitrary power polynomial layers.";
    }
    adjust(alpha) {
        this.fields.forEach((_, p) => {
            elementWise(this.fields[p].dim, index => {
                this.fields[p].add_to(index, -alpha * this.dfields[p].get(index));
            });
        });
        delete this.dfields;
    }
    gradientLengthSquared() {
        return sumOverIndices(this.fields.length, ([p]) => sumOverIndices([outputSize, ...Array(p).fill(inputSize)], (index) => this.fields[p].get(index) ** 2));
    }
    nudge(alpha) {
        this.fields.forEach((_, p) => {
            elementWise(this.fields[p].dim, index => {
                this.fields[p].add_to(index, initRand(alpha));
            });
        });
    }
}

class KernelLayer extends Layer {
    constructor(inputSize, inputChannels, outputChannels, kernelSpecs, randomRange = 1) {
        super([...inputSize, inputChannels], null);

        this.kernelSpecs = kernelSpecs;

        this.default2d('kernelSize', this.kernelSpecs.kernelSize);
        this.default2d('padding', 0);
        this.default2d('innerPadding', 0);
        this.default2d('stride', 1);

        this.kernelSpecs.inputSize = inputSize;
        this.kernelSpecs.trueInputSize = [0, 1].map(i => 2 * this.kernelSpecs.padding[i] + this.kernelSpecs.inputSize[i] + this.kernelSpecs.innerPadding[i] * (this.kernelSpecs.inputSize[i] - 1));
        this.kernelSpecs.outputSize = [0, 1].map(i => 1 + Math.floor((this.kernelSpecs.trueInputSize[i] - this.kernelSpecs.kernelSize[i]) / this.kernelSpecs.stride[i]));

        this.outputSize = [...this.kernelSpecs.outputSize, outputChannels];

        this.kernels = Array(outputChannels).fill().map(() => Array(inputChannels).fill().map(() => new Kernel(this.kernelSpecs, randomRange)));
        this.bias = Array(outputChannels).fill().map(() => initRand(randomRange));
    }
    // helper function that adds default value to kernel specs, or if the value is a number then it copies it twice so that its 2 dimensional.
    // This allows users to theoretically define any size for any of these parameters
    default2d(attribute, defaultValue) {
        if (this.kernelSpecs[attribute] === undefined) this.kernelSpecs[attribute] = [defaultValue, defaultValue];
        if (typeof this.kernelSpecs[attribute] === 'number') this.kernelSpecs[attribute] = [this.kernelSpecs[attribute], this.kernelSpecs[attribute]];
    }
    passOne(input) {
        const outputtedKernelValues = this.kernels.map(outputs => outputs.map((kernel, inputChannel) => kernel.pass(input, inputChannel)));

        return elementWise(this.outputSize, ([ox, oy, i]) =>
            this.bias[i] +
            sumOverIndices([this.inputSize[2]], ([j]) => outputtedKernelValues[i][j].get([ox, oy]))
        );
    }
    initDerivatives() {
        this.dbias = zerosTensor([this.outputSize[2]]);
        this.kernels.forEach(kernelRow => kernelRow.forEach(kernel => {
            kernel.initDerivatives();
        }));
    }
    setOneDerivative(delta, batchIdx) {
        this.dbias = elementWise([outputChannels], ([i]) => sumOverIndices(processedKernelSpecs.outputSize, ([ox, oy]) => this.dbias.get(i) + delta.get([ox, oy, i])));

        this.kernels.forEach((kernelRow, outputChannel) => kernelRow.forEach((kernel, inputChannel) => {
            kernel.setOneDerivative(delta, this.lastInput[batchIdx], inputChannel, outputChannel);
        }));

        const newDelta = zerosTensor(this.inputSize);
        elementWise([this.inputSize[2]], ([j]) => this.kernels[0][0].inputOutputLoop(([ox, oy, r, s], kernelIndex) => {
            newDelta.add_to([r, s, j],
                sumOverIndices([this.outputSize[2]], ([i]) => delta.get([ox, oy, i]) * this.kernels[i][j].adjustment.get([ox, oy, r, s]))
            )
        }));
        return newDelta;
    }
    adjust(alpha) {
        for (let n = 0; n < outputChannels; n++) {
            this.bias[n] -= this.dbias.get(n) * alpha;
            for (let m = 0; m < inputChannels; m++) {
                this.kernels[n][m].adjust(alpha);
            }
        }
        delete this.dbias;
    }
    gradientLengthSquared() {
        return sumOverIndices([outputChannels], ([n]) => this.dbias.get(n) ** 2) + sumOverIndices([outputChannels, inputChannels], ([n, m]) => this.kernels[n][m].gradientLengthSquared());
    }
    nudge(alpha) {
        for (let n = 0; n < outputChannels; n++) {
            this.bias[n] += mutation * (Math.random() * 2 - 1);
            for (let m = 0; m < inputChannels; m++) {
                this.kernels[n][m].nudge(alpha);
            }
        }
    }
}

class Kernel {
    constructor(kernelSpecs, randomRange = 1) {
        for (const param in kernelSpecs) {
            this[param] = kernelSpecs[param];
        }
        this.kernel = randomTensor(this.kernelSize, randomRange);
    }
    // helper function that loops over all inputs and outputs, but only on sections that would be affected by the kernel. This increases the speed by a lot over just having a regular elementWise()
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
    }

    pass(input, inputChannel) {
        const output = zerosTensor(this.outputSize);
        this.inputOutputLoop(([ox, oy, r, s], kernelIndex) => {
            output.add_to([ox, oy],
                this.kernel.get(kernelIndex) *
                input.get([r, s, inputChannel]));
        });
        return output;
    }
    initDerivatives() {
        this.dkernel = zerosTensor(this.kernelSize);
    }
    setOneDerivative(delta, input, inputChannel, outputChannel) {
        this.adjustment = zerosTensor([...this.outputSize, ...this.inputSize]);
        this.inputOutputLoop(([ox, oy, r, s], kernelIndex) => {
            this.dkernel.add_to(kernelIndex, delta.get([ox, oy, outputChannel]) * input.get([r, s, inputChannel]));
            this.adjustment.set([ox, oy, r, s], this.kernel.get(kernelIndex));
        });
    }
    adjust(alpha) {
        elementWise(this.kernelSize, ([x, y]) => this.kernel.add_to([x, y], -alpha * this.dkernel.get([x, y])));
        delete this.dkernel;
    }
    gradientLengthSquared() {
        return sumOverIndices(this.kernelSize, ([x, y]) => this.dkernel.get([x, y]) ** 2);
    }
    nudge(alpha) {
        elementWise(this.kernelSize, ([x, y]) => {
            this.kernel.add_to([x, y], initRand(alpha));
        });
    }
}

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