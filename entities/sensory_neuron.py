class SensoryNeuron:

    def __init__(self,
                 learning_rate: float, 
                 value: int, 
                 weight: float):
        self._learning_rate = learning_rate
        self._value = value
        self._weight = weight

    @property
    def value(self) -> int:
        return self._value
    
    @property
    def weight(self) -> float:
        return self._weight
    
    def update_weight(self, target: int) -> None:
        self._weight = self._weight + (self._learning_rate * target * self._value)