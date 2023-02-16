const { LinearNeuralNet } = require("./neuralnet");

let canvas, ctx;

window.onload = function () {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    init();
    const start = () => {
        frame();
        setTimeout(start, 1);
    };
    start();
};

const init = () => {

};

const frame = () => {

};

const draw = () => {

};

const board = Array(HEIGHT).fill().map(() => Array(WIDTH));

const makeState = (board, turn) => {
    return [...(turn ? [1, 0] : [0, 1]), ...board.flat().map(space => {
        switch (space) {
            case 'X': return [1, 0, 0];
            case 'O': return [0, 1, 0];
            default: return [0, 0, 1];
        }
    }).flat()];
}

const estimator = new LinearNeuralNet([29, 3], 1, ReLU, softmax);

const pickMove = () => {
    const state = makeState(board);
    const estimate = estimator.pass([state]);
}

