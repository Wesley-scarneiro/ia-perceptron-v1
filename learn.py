from configs.parameters import Parameters
from configs.learning_data import LearningData
from entities.perceptron import Perceptron
import logging

class Learn:

    def __init__(self, data: list[LearningData]):
        self.__data = data
        self.__perceptron = self._create_perceptron()
        self.__iterations_total = 0

    def _create_perceptron(self) -> Perceptron:
        logging.info("-- Creating percepton --")
        percepton = Perceptron(Parameters.inputs_total)
        logging.info(f"\t{percepton}")
        return percepton
    
    def start_learning(self) -> None:
        logging.info("-- Starting learning --")
        while(True):
            self.__iterations_total += 1
            updated_weights = False
            logging.info(f"\t-- Iterating data --")
            for data in self.__data:
                logging.info(f"\t\t- {data}")
                output = self.__perceptron.output(data.inputs)
                logging.info(f"\t\t- output={output}")
                if output != data.target:
                    logging.info(f"\t\t-- Update weights --")
                    self.__perceptron.update_synaptic_weights(data.target)
                    logging.info(f"\t\t\t- SensoryNeurons={self.__perceptron.sensory_neurons}")
                    updated_weights = True
                    break
            logging.info(f"\t-- Finished iteration --")
            if (not updated_weights):
                break
        logging.info("-- Finished learning --")

    def test_learning_perceptron(self) -> None:
        logging.info("-- Test learning perceptron --")
        for data in self.__data:
            output = self.__perceptron.output(data.inputs)
            logging.info(f"Input = {data.inputs} | Output = {output} | Correct? {output == data.target}")

    def learning_log(self) -> None:
        logging.info("-- Learning log --")
        logging.info(f"\t- {self.__perceptron}")
        logging.info(f"\t- Iterations total={self.__iterations_total}")