export const ANN_DEFAULT_BIAS = 0.001;

export function round_arr (arr = []){
	return arr.map((x) => Math.round(x));
}

export function round_equal (x = [], y = []){
	const x_round = round_arr(x);
	const y_round = round_arr(y);
	for (let i = 0; i < x_round.length; i++) {
		const _x = x_round[i];
		const _y = y_round[i];
		if (_x !== _y) {
			return false;
		}
	}
	return true;
}

export class ArtificialNeuralNetwork {
	constructor (
		weights = [],
		biases = [],
		layers_config = [
			1,
			1,
			1,
		],
	) {
		// [ layer1: [ node1_weights=[], node2_weights=[] ], layer2: [ node1_weights=[], node2_weights=[] ] ]
		this.weights = weights;
		if (!this.weights || this.weights.length === 0) {
			this.set_random_weights(layers_config);
		}

		// [ layer1: [ node1=0, node2=1 ], layer2: [ node1=0, node2=1 ] ]
		this.layers = [];
		this.reset_layers();

		// [ layer1: [ node1_bias=0.01, node2_bias=0.01 ], layer2: [ node1_bias=0.01, node2_bias=0.01 ] ]
		this.biases = biases;
		if (!this.biases || this.biases.length === 0) {
			this.set_default_biases();
		}
	}

	set_random_weights (
		layer_count_config = [
			1,
			1,
			1,
		],
	) {
		// [ layer1: [ node1_weights=[], node2_weights=[] ], layer2: [ node1_weights=[], node2_weights=[] ] ]
		this.weights = [];
		for (let i = 0; i + 1 < layer_count_config.length; i++) {
			this.weights[i] = [];

			const layer_size = layer_count_config[i];

			const next_layer_size = layer_count_config[i + 1];

			for (let node = 0; node < layer_size; node++) {
				this.weights[i][node] = [];
				for (let nl = 0; nl < next_layer_size; nl++) {
					this.weights[i][node][nl] = Math.random();
				}
			}
		}
	}

	set_default_biases (default_value = 0.001) {
		// [ layer1: [ node1_bias=0.01, node2_bias=0.01 ], layer2: [ node1_bias=0.01, node2_bias=0.01 ] ]
		this.biases = [];
		for (let i = 0; i < this.layers.length; i++) {
			this.biases[i] = [];
			const layer = this.layers[i];
			for (let node = 0; node < layer.length; node++) {
				if (i == 0 || i == this.layers.length - 1) {
					this.biases[i][node] = 0;
				}
				else {
					this.biases[i][node] = default_value;
				}
			}
		}
	}

	reset_layers () {
		// [ layer1: [ node1=0, node2=1 ], layer2: [ node1=0, node2=1 ]]
		this.layers = [];
		let next_layer_size = 0;
		for (let i = 0; i < this.weights.length; i++) {
			const weight = this.weights[i];
			this.layers[i] = [];
			for (let node = 0; node < weight.length; node++) {
				this.layers[i][node] = 0;
			}
			const first_w = weight[0];
			if (first_w !== undefined) {
				next_layer_size = first_w.length;
			}
		}
		let i = this.weights.length;
		this.layers[i] = [];
		for (let node = 0; node < next_layer_size; node++) {
			this.layers[i][node] = 0;
		}
	}

	sigmoid_x (x = 0) {
		return 1 / (1 + Math.exp(-x));
	}

	deriv_sigmoid (activated_output = 0) {
		return activated_output * (1 - activated_output);
	}

	de_sigmoid (sigmoid_val = 0) {
		return -Math.log((1 - sigmoid_val) / sigmoid_val);
	}

	deactivate_array_output (arr = [], biases = []) {
		const deactivated = [];
		for (let i = 0; i < arr.length; i++) {
			const activated_node = arr[i];
			const bias = biases[i];
			deactivated[i] = de_sigmoid(activated_node) - bias;
		}
		return deactivated;
	}

	activate_array_output (arr = [], biases = []) {
		const activated = [];
		for (let i = 0; i < arr.length; i++) {
			const node = arr[i];
			let bias = biases[i];
			if (bias === undefined) {
				bias = 0;
			}
			const node_p_biase = node + bias;
			activated[i] = this.sigmoid_x(node_p_biase);
		}
		return activated;
	}

	feed_forward (input = []) {
		this.reset_layers();
		this.layers[0] = input;
		for (let i = 0; i + 1 < this.layers.length; i++) {
			const next_layer = [];
			const layer = this.layers[i];
			const weight = this.weights[i];
			const next_bias = this.biases[i + 1];

			for (let n = 0; n < layer.length; n++) {
				const node = layer[n];
				const node_weights = weight[n];

				for (let nw = 0; nw < node_weights.length; nw++) {
					const node_weight = node_weights[nw];
					if (next_layer[nw] === undefined) {
						next_layer[nw] = 0;
					}
					next_layer[nw] += node_weight * node;
				}
			}
			this.layers[i + 1] = this.activate_array_output(next_layer.slice(), next_bias);
		}
	}

	back_propagate (target = [], learning_rate = 0.001) {
		const last_index = this.layers.length - 1;
		let deriv_error = [];
		let deriv_output = [];
		let prev_layer_delta = [];

		// set derivatives for output layer
		for (let i = 0; i < this.layers[last_index].length; i++) {
			const node = this.layers[last_index][i];
			deriv_error[i] = node - target[i];
			deriv_output[i] = this.deriv_sigmoid(node);
		}

		// loop layers from last layer to second layer
		// did not include first layer since it is the input layer
		for (let i = last_index; i >= 1; i--) {
			const layer = this.layers[i];

			const next_layer = this.layers[i - 1];
			const weight = this.weights[i - 1];

			const biases = this.biases[i];
			const new_weights = [];
			const new_deriv_error = [];
			const new_deriv_output = [];

			// set deltas of the current layer of nodes
			for (let n = 0; n < layer.length; n++) {
				prev_layer_delta[n] = deriv_error[n] * deriv_output[n];
				if (i !== this.biases.length - 1) {
					// const bias = biases[n];
					biases[n] -= learning_rate * prev_layer_delta[n];
				}
			}

			// loop through the next(preceding) layer using deltas of the current layer
			for (let n = 0; n < next_layer.length; n++) {
				const node = next_layer[n];
				const node_weights = weight[n];
				const new_node_weights = node_weights.slice();

				// compute new derivatives for hidden layers
				new_deriv_error[n] = 0;
				new_deriv_output[n] = this.deriv_sigmoid(node);

				for (let nw = 0; nw < node_weights.length; nw++) {
					const delta = prev_layer_delta[nw];
					const node_weight = node_weights[nw];

					// compute the changes need for new weights
					new_node_weights[nw] -= learning_rate * delta * node;

					new_deriv_error[n] += node_weight * delta;
				}
				new_weights[n] = new_node_weights.slice();
			}

			// save the new weights computed
			this.weights[i - 1] = new_weights.slice();

			// set current layer's derivatives to be used for computing the deltas
			deriv_error = new_deriv_error.slice();
			deriv_output = new_deriv_output.slice();
			// for (let n = 0; n < layer.length; n++) {
			// 	const delta = deriv_error[n] * deriv_output[n];

			// }
		}
	}

	predict (input = []) {
		this.feed_forward(input);
		return this.layers[this.layers.length - 1].slice();
	}

	train (
		training_data = [
			{ input: [], output: [] },
		],
		learn_rate = 0.001,
		max_training_loop = 10000,
		log_per_loop = 100000,
	) {
		for (let i = 0; i < max_training_loop; i++) {
			let score = 0;
			for (const data of training_data) {
				// this.feed_forward(data.input);
				// const output = this.layers[this.layers.length - 1];
				const output = this.predict(data.input);
				if (round_equal(output, data.output)) {
					score += 1;
					// continue;
				}
				this.back_propagate(data.output, learn_rate);
			}
			if (score == training_data.length) {
				// console.log('score', score);
				// console.log('iterations', i);
				break;
			}
			// if (i % log_per_loop == 0) {
			// 	console.log('score', score);
			// }
		}
	}

	hard_train (
		training_data = [
			{ input: [], output: [] },
		],
		learn_rate = 0.001,
		training_loop = 10000,
		log_per_loop = 10000,
	) {
		for (let i = 0; i < training_loop; i++) {
			this.train(training_data, learn_rate, 1, log_per_loop);
		}
	}

	freeze () {
		return JSON.parse(JSON.stringify({ w: this.weights, b: this.biases }));
	}
}
