from random import uniform
from configs.parameters import Parameters
from .sensory_neuron import SensoryNeuron
from .activation_function import ActivationFunction

class Perceptron:

    __id = 0

    def __init__(self,
            bias: SensoryNeuron,
            functionType: int):
        self.__bias = bias
        self.__functionType = functionType
        self.__sensory_neurons = []
        Perceptron.__id += 1
        self.__id = Perceptron.__id

    @property
    def sensory_neurons(self) -> list[SensoryNeuron]:
        return self.__sensory_neurons
        
    def _input_function(self) -> float:
        value = self.__bias.weight
        for sensory_neuron in self.__sensory_neurons:
            value += sensory_neuron.value * sensory_neuron.weight
        return value
    
    def _step_function(self, value: float) -> int:
        return ActivationFunction.step_function_binary(value)

    def _receive_values(self, list_values: list[int]) -> None:
        if not self.__sensory_neurons:
            for value in list_values:
                self.__sensory_neurons.append(SensoryNeuron(value, Parameters._weight_random()))
        else:
            for value, sensory_neuron in zip(list_values, self.__sensory_neurons):
                sensory_neuron.value = value

    def output(self, list_values: list[int]) -> int:
        self._receive_values(list_values)
        value = self._step_function(self._input_function())
        return value
    
    def update_synaptic_weights(self, target: int) -> None:
        self.__bias.update_weight(target)
        for sensory_neuron in self.__sensory_neurons:
            sensory_neuron.update_weight(target)

    def __str__(self):
        string = f"Perceptron=[Id={self.__id}, FunctionType={self.__functionType}, Bias={self.__bias}, SensoryNeurons={self.__sensory_neurons}]"
        return string