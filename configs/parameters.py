from random import uniform

class Parameters:

    learning_rate = 0.0
    bias_value = 1
    
    @staticmethod
    def _weight_random() -> float:
        return round(uniform(0, 1), 3)