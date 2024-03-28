from sensory_neuron import SensoryNeuron
from activation_function import ActivationFunction

class Perceptron:

    def __init__(self,
            sensory_neurons: list[SensoryNeuron], 
            bias: SensoryNeuron,
            functionType: int,
            target: int):
        self._sensory_neurons = sensory_neurons
        self._bias = bias
        self._functionType = functionType
        self._target = target
    
    @property
    def target(self) -> int:
        return self._target
    
    def _input_function(self) -> float:
        value = self._bias.value * self._bias._weight
        for sensory_neuron in self._sensory_neurons:
            value += sensory_neuron.value * sensory_neuron.weight
        return value
    
    def _step_function(self, value: float) -> int:
        if self._functionType:
            return ActivationFunction.step_function_binary(value)
        return ActivationFunction.step_function_bipolar(value)
    
    def output(self) -> int:
        value = self._step_function(self._input_function())
        return value
    
    def update_synaptic_weights(self) -> None:
        self._bias.update_weight()
        for sensory_neuron in self._sensory_neurons:
            sensory_neuron.update_weight()
