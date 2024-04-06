class LearningData:

    def __init__(self, inputs: list[int], target: int):
        self.__inputs = inputs
        self.__target = target
    
    @property
    def inputs(self) -> list[int]:
        return self.__inputs
    
    @property
    def target(self) -> int:
        return self.__target
    
    def __str__(self):
        return f"LearningData=[Inputs={self.__inputs}, Target={self.__target}]"