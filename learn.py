from configs.parameters import Parameters
from configs.learning_data import LearningData
from entities.sensory_neuron import SensoryNeuron
from entities.perceptron import Perceptron
import logging

class Learn:

    def __init__(self, data: list[LearningData]):
        self.__data = data
        self.__perceptron = self._create_perceptron()

    def _create_perceptron(self):
        logging.info("-- Creating percepton --")
        bias = SensoryNeuron(Parameters.bias_value, Parameters._weight_random())
        percepton = Perceptron(bias)
        logging.info(f"\t{percepton}")
        return percepton
    
    def start_learning(self) -> None:
        logging.info("-- Starting learning --")
        while(True):
            updated_weights = False
            logging.info(f"\t-- Iterating data --")
            for data in self.__data:
                logging.info(f"\t\t- {data}")
                output = self.__perceptron.output(data.inputs)
                logging.info(f"\t\t- output={output}")
                if output != data.target:
                    logging.info(f"\t\t-- Update weights --")
                    self.__perceptron.update_synaptic_weights(data.target)
                    logging.info(f"\t\t\t- Sensory neurons={self.__perceptron.sensory_neurons}")
                    updated_weights = True
                    break
            logging.info(f"\t-- Finished iteration --")
            if (not updated_weights):
                break
        logging.info("-- Finished learning -- \n")

    def test_learning_perceptron(self):
        logging.info("-- Test learning perceptron --")
        for data in self.__data:
            output = self.__perceptron.output(data.inputs)
            logging.info(f"Input = {data.inputs} | Output = {output} | Correct? {output == data.target}")

