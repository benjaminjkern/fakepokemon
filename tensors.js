const newTensor = (dim, data) => {
    const size = dimSize(dim);
    if (size !== data.length) throw `${dim} (size: ${size}) does not match data size: ${data.length}!`;
    for (const d of data) {
        if (typeof d !== 'number') {
            throw `All members in a tensor must be numbers! (Received: ${d})`;
        }
    }
    const tensor = zerosTensor(dim);
    tensor.data = data;
    return tensor;
};

const newSparseTensor = (dim, data) => {
    const tensor = zerosTensor(dim);
    tensor.data = data;
    return tensor;
};

const randomTensor = (dim, range = 1, offset = 0) => {
    const tensor = zerosTensor(dim);
    tensor.isRandom = [range, offset];
    return tensor;
}

const zerosTensor = (dim) => {
    const data = Array(dimSize(dim));
    data[data.length - 1] = undefined;
    return {
        dim,
        data,
        get(index) {
            if (typeof index !== 'object') return this.get([index]);
            return this.get_byDataIdx(createIdx(index, this.dim));
        },
        get_byDataIdx(dataIdx) {
            if (this.data[dataIdx] === undefined) {
                if (this.isRandom) {
                    this.data[dataIdx] = this.isRandom[0] * (Math.random() * 2 - 1) + this.isRandom[1];
                } else {
                    return 0;
                }
            }
            return this.data[dataIdx];
        },
        get_default(index, default_value) {
            if (typeof index !== 'object') return this.get([index]);
            try {
                createIdx(index, this.dim);
            } catch (e) {
                return default_value;
            }
            return this.get(index);
        },
        add_to(index, value) {
            const dataIdx = createIdx(index, this.dim);
            this.set_byDataIdx(dataIdx, this.get_byDataIdx(dataIdx) + value);
        },
        set(index, value) {
            if (typeof index !== 'object') return this.set([index], value);
            return this.set_byDataIdx(createIdx(index, this.dim), value);
        },
        set_byDataIdx(dataIdx, value) {
            if (typeof value !== 'number') throw `All members in a tensor must be numbers! (Received: ${value})!`;
            this.data[dataIdx] = value;
        },
        map(mapFunc) {
            return newSparseTensor(this.dim, this.data.map(mapFunc));
        }
    };
}

const createIdx = (index, dim) => {
    if (index.length !== dim.length) throw `Cannot cast ${index} to dimension ${dim}!`;
    for (const [i, idx] of index.entries()) {
        if (idx < 0) throw `${idx} is out of range (min: 0)!`;
        if (idx >= dim[i]) throw `${idx} is out of range (max: ${dim[i] - 1})!`;
        if (idx % 1 !== 0) throw `${idx} is not an integer!`;
    }
    return index.reduce((p, idx, i) => p * dim[i] + idx, 0);
}

const recreateIdx = (i, dim) => {
    const index = Array(dim.length - 1);
    for (let idx = dim.length - 1; idx >= 0; idx--) {
        index[idx] = i % dim[idx];
        i = Math.floor(i / dim[idx]);
    }
    return index;
}

const addToIdx = (index, dim) => {
    let d = index.length - 1;
    while (d >= 0) {
        index[d]++;
        if (index[d] >= dim[d]) {
            index[d] = 0;
            d--;
            continue;
        }
        return index;
    }
    return index;
}

const dimSize = (dim) => dim.reduce((p, c) => p * c, 1);

const elementWise = (dim, func) => {
    const tensor = zerosTensor(dim);
    const size = dimSize(dim);
    let index = recreateIdx(0, dim);
    for (let i = 0; i < size; i++) {
        const result = func(index);
        if (result) tensor.set_byDataIdx(i, result);
        index = addToIdx(index, dim);
    }
    return tensor;
}

const sumOverIndices = (dim, func) => {
    let sum = 0;
    const size = dimSize(dim);
    let index = recreateIdx(0, dim);
    for (let i = 0; i < size; i++) {
        sum += func(index);
        index = addToIdx(index, dim);
    }
    return sum;
}