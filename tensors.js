const newTensor = (dim, data) => {
    const size = dimSize(dim);
    if (size !== data.length) throw `${dim} (size: ${size}) does not match data size: ${data.length}!`;
    if (data.some(d => !(d instanceof Number))) throw `All members in a tensor must be numbers!`;
    const tensor = zerosTensor(dim);
    tensor.data = data;
    return tensor;
};

const randomTensor = (dim, range = 1) => {
    const tensor = zerosTensor(dim);
    tensor.isRandom = range;
    return tensor;
}

const zerosTensor = (dim) => {
    const data = Array(dimSize(dim));
    data[data.length - 1] = undefined;
    return {
        dim,
        data,
        get(index) {
            if (!(index instanceof Array)) return this.get([index]);
            const i = createIdx(index, this.dim);
            if (this.data[i] === undefined) {
                if (this.isRandom) {
                    this.data[i] = this.isRandom * (Math.random() * 2 - 1);
                } else this.data[i] = 0;
            }
            return this.data[i];
        },
        set(index, value) {
            if (typeof index !== 'object') return this.set([index], value);
            return this.set_byDataIdx(createIdx(index, this.dim), value);
        },
        set_byDataIdx(dataIdx, value) {
            if (typeof value !== 'number') throw `All members in a tensor must be numbers (received: ${value})!`;
            return this.data[dataIdx] = value;
        },
        map(mapFunc) {
            return newTensor(this.dim, data.map(mapFunc));
        }
    };
}

const createIdx = (index, dim) => {
    if (index.length !== dim.length) throw `Cannot cast ${index} to dimension ${dim}!`;
    for (const [i, idx] of index.entries()) {
        if (idx < 0) throw `${idx} is out of range (min: 0)!`;
        if (idx >= dim[i]) throw `${idx} is out of range (max: ${dim[i] - 1})!`;
    }
    return index.reduce((p, idx, i) => p * dim[i] + idx, 0);
}

const recreateIdx = (i, dim) => {
    const newDim = [...dim].reverse();
    return newDim.reduce((p, d) => [Math.floor(p[0] / d), [p[0] % d, ...p[1]]], [i, []])[1];
}

const dimSize = (dim) => dim.reduce((p, c) => p * c, 1);

const elementWise = (dim, func) => {
    const tensor = zerosTensor(dim);
    const size = dimSize(dim);
    for (let i = 0; i < size; i++) {
        const index = recreateIdx(i, dim);
        tensor.set_byDataIdx(i, func(index));
    }
    return tensor;
}

const sumOverIndices = (dim, func) => {
    let sum = 0;
    const size = dimSize(dim);
    for (let i = 0; i < size; i++) {
        const index = recreateIdx(i, dim);
        sum += func(index);
    }
    return sum;
}

// const M = 2;
// const N = 10;

// const a = randomTensor([M, N, N]);
// const b = randomTensor([M, N]);
// const c = randomTensor([M]);
// const x = randomTensor([N]);

// const output = elementWise([M], ([i]) =>
//     sumOverIndices([N, N], ([j, k]) => a.get([i, j, k]) * x.get(j) * x.get(k)) +
//     sumOverIndices([N], ([j]) => b.get([i, j]) * x.get(j)) +
//     c.get(i)
// );

// console.log(x);
// console.log(output);
// console.log(elementWise([M], ([i]) => 2 * c.get(i)))
// console.log(sumOverIndices([M], ([i]) => c.get(i)));