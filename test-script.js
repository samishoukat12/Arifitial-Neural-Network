const fronzen_key = 'frozen';

// check if there is a fronzen neural network in the local storage and set to weights and biases
// if there is none then set weights and biases to empty array

console.log("key", fronzen_key);
const { w, b } = JSON.parse(localStorage.getItem(fronzen_key)) || {
	w: [],
	b: [],
};
// const w = fronzen_brain.w,
// 	b = fronzen_brain.b;
// console.log('fronzen_brain', fronzen_brain);

import { ArtificialNeuralNetwork } from './ann.js';

const layer_config = [
	2, // nodes in input layer
	5, // nodes in hidden layer 1
	5, // nodes in hidden layer 2
	1, // nodes in ouput layer
];

const ann = new ArtificialNeuralNetwork(w, b, layer_config);

// test XOR Dataset for Ann to learn
const x_or_data = [
	{
		input: [
			1,
			0,
		],
		output: [
			1,
		],
	},
	{
		input: [
			0,
			1,
		],
		output: [
			1,
		],
	},
	{
		input: [
			1,
			1,
		],
		output: [
			0,
		],
	},
	{
		input: [
			0,
			0,
		],
		output: [
			0,
		],
	},
];

function predict_xor_dataset() {
	for (const data of x_or_data) {
		console.log('Expected output', data.output);
		const output = ann.predict(data.input);
		console.log('ANN output', output);
	}
}

// prediction before training
console.log('Predictions Before Training');
predict_xor_dataset();

console.log('ann', JSON.parse(JSON.stringify(ann)));
const learn_rate = 0.01;
const max_training_loop = 1000000;
const log_per_loop = 10000;

ann.train(x_or_data, learn_rate, max_training_loop, log_per_loop);

// predictions after Training
console.log('Predictions After Training');
predict_xor_dataset();

// if (!localStorage.getItem(fronzen_key)) {
// 	ann.hard_train(x_or_data, 0.01, 500000, 10000);
// }

// // save in the local storage
// localStorage.setItem(fronzen_key, JSON.stringify(ann.freeze()));
// console.log('ann', JSON.parse(JSON.stringify(ann)));
