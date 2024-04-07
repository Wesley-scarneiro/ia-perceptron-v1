from configs.parameters import Parameters


class SensoryNeuron:

    __id = 0
    
    def __init__(self,
                 value: int, 
                 weight: float):
        self.__value = value
        self.__weight = weight
        SensoryNeuron.__id += 1
        self.__id = SensoryNeuron.__id

    @property
    def value(self) -> int:
        return self.__value
    
    @value.setter
    def value(self, value) -> int:
        self.__value = value
    
    @property
    def weight(self) -> float:
        return self.__weight
    
    @staticmethod
    def get_learning_rate() -> float:
        return SensoryNeuron.__learning_rate
    
    @staticmethod
    def set_learning_rate(value) -> None:
        SensoryNeuron.__learning_rate = value
    
    def update_weight(self, target: int) -> None:
        self.__weight += round(Parameters.learning_rate * target * self.__value, 3)
    
    def __str__(self):
        string = f"[Id={self.__id}, Value={self.__value}, Weight={self.__weight}]"
        return string
    
    def __repr__(self):
        string = f"[Id={self.__id}, Value={self.__value}, Weight={self.__weight}]"
        return string