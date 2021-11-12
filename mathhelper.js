const transpose = (a) => a[0].map((_, col) => a.map((row) => row[col]));

// const transpose = (A) => {
//     const ASize = size(A);
//     if (ASize.length === 1) return transpose([A]);
//     if (ASize.length % 2 !== 0) throw "INVALID DIMENSIONS";
//     if (ASize.length === 0) return A;
//     return A[0].map((_, i) => A.map(row => transpose(row[i])));
// }

const resize = (a, ...newSize) => unflatten(flatten(a), ...newSize);

const unflatten = (a, ...newSize) => {
    if (!newSize.length) return a[0];
    if (newSize[0] === 0) return [];
    return [unflatten(a.slice(0, a.length / newSize[0]), ...newSize.slice(1)),
        ...unflatten(a.slice(a.length / newSize[0]), ...newSize.map((x, i) => i === 0 ? x - 1 : x))
    ];
}

const flatten = (a) => {
    if (!a.length) return [a];
    return a.reduce((p, c) => [...p, ...flatten(c)], []);
}

const matadd = (a, b) => {
    if (!a.length && !b.length) return a + b;
    if (a.length == b.length) return a.map((x, i) => matadd(x, b[i]));
    throw "Mismatched size";
}

// theoretically works on any size tensor but is slow ?
const matmult = (a, b) => {
    if (!a.length && !b.length) return a * b;
    if (!a.length) return [];
    if (a[0].length == b.length) return [b[0].map((_, bcol) => dotproduct(a[0], b.map(row => row[bcol]))), ...matmult(a.slice(1), b)];
    throw "Mismatched size";
}

const dotproduct = (a, b) => {
    if (a.length == 1 && b.length == 1) return matmult(a[0], b[0]);
    if (a.length == b.length) return matadd(matmult(a[0], b[0]), dotproduct(a.slice(1), b.slice(1)));
    throw "Mismatched size";
}

const zeros = (...sizes) => {
    if (!sizes.length) return 0;
    return Array(sizes[0]).fill().map(() => zeros(...sizes.slice(1)));
}

const size = (a) => {
    if (!a.length) return [];
    return [a.length, ...size(a[0])];
}



const quadratic = (a, b, c) => {
    let disc = b ** 2 - 4 * a * c;
    if (disc < 0 || a === 0) return [undefined];
    disc = Math.sqrt(disc);
    return [(-b - disc) / 2 / a, (-b + disc) / 2 / a];
}

const crossProduct = (u, v) => {
    if (u.length !== 3 || v.length !== 3) throw "Must use 2 3-dimensional vectors!";
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
}


const distSquared = ({ pos: pos1 }, { pos: pos2 }) => {
    const a = sub(pos2, pos1);
    return dot(a, a);
}

const rref = (A) => {
    if (size(A).length !== 2) throw "rref is not supported for non-matrices";
    const newA = deepCopy(A);
    for (let rowNum = 0; rowNum < A.length; rowNum++) {
        const topRow = newA[rowNum];
        const col = firstNonZeroIndex(topRow);
        if (col === -1) continue;

        newA[rowNum] = mult(topRow, 1 / topRow[col]);
        newA.forEach((row, i) => {
            if (i === rowNum) return;
            if (row[col] === 0) return;
            newA[i] = sub(row, mult(newA[rowNum], row[col]));
        });
    }

    return newA;
}

// assumes two matrices
const matMult = (A, B) => {
    if (!B[0].length)
        return A.map(row => dot(row, B));
    if (A[0].length !== B.length) throw `INVALID [${A.length}x${A[0].length}]*[${B.length}x${B[0].length}]`;

    const answer = Array(A.length).fill().map(() => Array(B[0].length).fill(0));
    for (let y = 0; y < A.length; y++) {
        for (let x = 0; x < B[0].length; x++) {
            for (let z = 0; z < B.length; z++) {
                answer[y][x] += A[y][z] * B[z][x];
            }
        }
    }
    return answer;
}

const firstNonZeroIndex = (list, i = 0) => {
    if (list.length === 0) return -1;
    if (list[i] === 0) return firstNonZeroIndex(list, i + 1);
    return i;
}

const project = (from, to) => mult(dot(from, to) / dot(to, to), to);

const unit = (vec) => mult(1 / length(vec), vec);

const lengthSquared = (vector) => dot(vector, vector);

const length = (vector) => Math.sqrt(lengthSquared(vector));

const rotationMatrix = (theta, u) => {
    const cos = Math.cos(theta);
    const sin = Math.sin(theta);
    if (u === undefined)
        return [
            [Math.cos(theta), -Math.sin(theta)],
            [Math.sin(theta), Math.cos(theta)]
        ];
    if (lengthSquared(u) !== 1) {
        u = mult(u, 1 / length(u));
    }
    return [
        [cos + (1 - cos) * u[0] ** 2, u[0] * u[1] * (1 - cos) - u[2] * sin, u[0] * u[2] * (1 - cos) + u[1] * sin],
        [u[0] * u[1] * (1 - cos) + u[2] * sin, cos + (1 - cos) * u[1] ** 2, u[1] * u[2] * (1 - cos) - u[0] * sin],
        [u[0] * u[2] * (1 - cos) - u[1] * sin, u[1] * u[2] * (1 - cos) + u[0] * sin, cos + (1 - cos) * u[2] ** 2],
    ];
}

// const size = (A) => {
//     if (!A.length) return [];
//     const first = size(A[0]);
//     if (A.slice(1).every(x => deepEquals(size(x), first))) return [A.length, ...first];
//     throw "INCONSISTENT DIMENSIONS";
// }

const deepCopy = (A) => {
    if (typeof A !== 'object') return A;
    if (A.length !== undefined) return A.map(deepCopy);
    return Object.keys(A).reduce((p, key) => ({...p, [key]: deepCopy(A[key]) }), {});
}

const deepEquals = (A, B) => {
    if (typeof A !== typeof B) return false;
    if (typeof A !== 'object') return A === B;
    if (A.length !== B.length) return false;
    if (A.length) return A.every((a, i) => deepEquals(a, B[i]));

    const AKeys = Object.keys(A);
    const BKeys = Object.keys(B);
    if (AKeys.length !== BKeys.length) return false;
    return AKeys.every(key => deepEquals(A[key], B[key]));
}

const dot = (A, B) => {
    if (A.length !== B.length) throw "INVALID DIMENSIONS";
    if (A.length === 1) return mult(A[0], B[0]);
    return A.reduce((p, c, i) => p ? add(p, mult(c, B[i])) : mult(c, B[i]), undefined);
};

const add = (A, B, validated = false) => {
    const validate = validated || deepEquals(size(A), size(B));
    if (!validate) throw "INVALID DIMENSIONS";
    if (A.length === undefined) return A + B;
    if (A.length === 0) return [];
    return [add(A[0], B[0], validate), ...add(A.slice(1), B.slice(1), validate)];
}

const neg = (A) => mult(-1, A);

const elemMult = (A, B) => {
    return A.map((a, i) => mult(a, B[i]));
}

const mult = (A, B) => {
    const ASize = size(A);
    const BSize = size(B);

    if (ASize.length === 0) {
        if (BSize.length === 0) return A * B;
        return B.map(b => mult(A, b));
    }
    if (BSize.length === 0) return mult(B, A);
    if (BSize.length === 1) return mult(A, transpose(B));
    if (ASize.length < 2) throw "INVALID DIMENSIONS";
    if (ASize[1] !== BSize[0]) throw "INVALID DIMENSIONS";
    return A.map(row => transpose(B).map(col => dot(row, col)));
}

const sub = (A, B) => add(A, neg(B));

const element = (A) => {
    const ASize = size(A);
    if (ASize.length === 0) return A;
    if (ASize.every(dim => dim === 1)) return element(A[0]);
    throw "INVALID DIMENSIONS";
}

const concat = (A, B) => transpose([...transpose(A), ...transpose(B)]);

const identity = (s) => Array(s).fill().map((_, i) => Array(s).fill().map((_, j) => i == j ? 1 : 0));

const print = (A) => {
    console.log(A.map(row => row.map(num => (Math.round(num * 100) / 100)).join('\t')).join('\n'), '\n');
}

// TODO: MAKE BETTER
const inverse = (A) => {
    if (ASize.length !== 2 || ASize[0] !== ASize[1]) throw "Must be a square matrix";
    transpose(transpose(rref(concat(A, identity(A.length)))).slice(A.length));
}


const initRand = (range = 1) => (Math.random() * 2 - 1) * range;

const rands = (sizes, range = 1) => {
    if (!sizes.length) return initRand(range);
    return Array(sizes[0]).fill().map(() => rands(sizes.slice(1), range));
}