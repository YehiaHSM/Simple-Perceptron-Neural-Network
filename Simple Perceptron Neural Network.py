import numpy as np

class NeuralNetwork:

    def __init__(self, learning_rate, threshold):
        # Seeding for random number generation
        self.learning_rate = learning_rate
        self.threshold = threshold
        np.random.seed(1)
        # Initialize weights: converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def step(self, x):
        # Applying the step function
        return 1 if x > float(self.threshold) else 0

    def train(self, training_inputs, training_outputs, training_iterations):
        # Training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # Siphon the training data via the neuron
            output = self.think(training_inputs)
            error = training_outputs - output
            # Performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.learning_rate)
            self.synaptic_weights += adjustments

    def think(self, inputs):
        # Passing the inputs via the neuron to get output
        # Converting values to floats
        inputs = inputs.astype(float)  # Convert input to float datatype
        output_in = np.sum(np.dot(inputs, self.synaptic_weights))  # Weighted sum of input
        output = self.step(output_in)
        return output

if __name__ == "__main__":
    # Initializing the neuron class
    learning_rate = float(input("Learning rate: "))
    threshold = float(input("Threshold: "))
    neural_network = NeuralNetwork(learning_rate, threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    # Training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1],
                                [0, 0, 0]])

    training_outputs = np.array([[1], [1], [1], [1], [0]])

    # Training taking place
    neural_network.train(training_inputs, training_outputs, 4)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    user_input_one = float(input("User Input One: "))
    user_input_two = float(input("User Input Two: "))
    user_input_three = float(input("User Input Three: "))

    print("Considering New Situation: ", user_input_one, user_input_two, user_input_three)
    print("New Output data: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print("Wow, we did it!")
