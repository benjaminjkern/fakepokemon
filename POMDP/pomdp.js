
const { zerosTensor, newTensor } = require('../tensors');
const { LinearNeuralNet, sigmoid } = require('../neuralnet');


const observations = ['start', 'isleft', 'isright', 'wasright', 'wasleft'];
const actions = ['openleft', 'openright', 'observe'];
const internalStates = ['tigerleft', 'tigerright'];

const rewards = {
    'tigerleft': {
        'openleft': -100,
        'openright': 1,
    },
    'tigerright': {
        'openleft': 1,
        'openright': -100,
    }
}

const obsProbs = {
    init: {
        'tigerleft': {
            'start': 1
        },
        'tigerright': {
            'start': 1
        }
    },
    'openleft': {
        'tigerleft': {
            'wasleft': 1
        },
        'tigerright': {
            'wasright': 1
        }
    },
    'openright': {
        'tigerleft': {
            'wasleft': 1
        },
        'tigerright': {
            'wasright': 1
        }
    },
    'observe': {
        'tigerleft': {
            'isleft': 0.75,
            'isright': 0.25
        },
        'tigerright': {
            'isleft': 0.25,
            'isright': 0.75
        }
    }
}

const stateProbs = {
    init: {
        init: {
            'tigerleft': 0.5,
            'tigerright': 0.5,
        },
    },
    'tigerleft': {
        'openleft': {
            'tigerleft': 0.5,
            'tigerright': 0.5,
        },
        'openright': {
            'tigerleft': 0.5,
            'tigerright': 0.5,
        },
        'observe': {
            'tigerleft': 1
        }
    },
    'tigerright': {
        'openleft': {
            'tigerleft': 0.5,
            'tigerright': 0.5,
        },
        'openright': {
            'tigerleft': 0.5,
            'tigerright': 0.5,
        },
        'observe': {
            'tigerright': 1
        }
    }
}

const environment = {
    internalState: 'init',
    init() {
        this.internalState = 'init';
        return this.transition('init');
    },
    transition(action) {
        const reward = !rewards[this.internalState] || !rewards[this.internalState][action] ? 0 : rewards[this.internalState][action];

        if (!stateProbs[this.internalState] || !stateProbs[this.internalState][action]) throw `Action ${action} is not available on state ${this.internalState}.`;
        this.internalState = pickOne(stateProbs[this.internalState][action]);

        if (!obsProbs[action] || !obsProbs[action][this.internalState]) throw "Something went wrong.";
        const nextObs = pickOne(obsProbs[action][this.internalState]);

        return [nextObs, reward];
    }
};

const run = (agent, environment, steps) => {
    let t = 0;
    agent.totalReward = 0;

    let [observation, reward] = environment.init();
    while (++t <= steps) {
        [observation, reward] = environment.transition(agent.choose(observation, reward));
        agent.totalReward += reward;
    }
    return agent.totalReward / steps;
}

const montecarlo = (num) => (runFunc) => {
    let n = 0;
    let sum = 0;
    let sumSquares = 0;
    while (++n <= num) {
        const value = runFunc();
        sum += value;
        sumSquares += value ** 2;
    }
    return [sum / num, Math.sqrt(sumSquares / num - (sum / num) ** 2)];
}

// Assume: Agent knows the set of all actions, states, and observations

const randomAgent = {
    choose: (observation, reward) => pickRand(actions)
}

const manualAgent = {
    choose(observation, reward) {
        if (['start', 'wasright', 'wasleft'].includes(observation)) return 'observe';
        if (observation == 'isleft') return 'openright';
        if (observation == 'isright') return 'openleft';
        throw observation;
    }
}

const memorySize = 2;

const rnnAgent = {
    memory: newTensor([memorySize], Array(memorySize).fill(0)),
    H: new LinearNeuralNet([memorySize + observations.length + actions.length + 1, 1000, memorySize], 1, sigmoid),
    Q: new LinearNeuralNet([memorySize + actions.length, 1000, 1]),
    // lastAction: 'init',
    choose(observation, reward = 0) {
        if (this.bestActionState) {
            const QDelta = this.Q.backProp([this.bestActionState], [newTensor([1], [reward])], 0.01);
            this.H.backProp(QDelta, QDelta, 0.01, QDelta);
        }
        const concat = newTensor([memorySize + observations.length + actions.length + 1],
            [...this.memory.data,
            ...oneHot(actions.indexOf(this.lastAction), actions.length),
            ...oneHot(observations.indexOf(observation), observations.length),
                reward]);
        this.memory = this.H.pass([concat])[0];

        const actionStates = actions.map(action => newTensor([memorySize + actions.length],
            [...this.memory.data, ...oneHot(actions.indexOf(action), actions.length)]));

        const QValues = this.Q.pass(actionStates).map(q => q.data[0]);

        const bestQIndex = argmax(QValues);

        this.bestActionState = actionStates[bestQIndex];

        console.log(QValues);

        return actions[bestQIndex];
    }
}

const argmax = (list) => {
    let max = -Number.MAX_SAFE_INTEGER;
    let idx = 0;
    for (const [i, item] of list.entries()) {
        if (item > max) {
            max = item;
            idx = i;
        }
    }
    return idx;
}

const oneHot = (id, length) => {
    const arr = Array(length).fill(0);
    if (id >= 0)
        arr[id] = 1;
    return arr;
}
const pickOne = (probDist) => {
    let r = Math.random();
    for (const key in probDist) {
        r -= probDist[key];
        if (r <= 0) return key;
    }
    throw 'Something went wrong!';
}

const pickRand = (list) => list[Math.floor(Math.random() * list.length)];

console.log(montecarlo(1)(() => run(rnnAgent, environment, 100)));

// console.log(rnnAgent.choose('start', 0));