class ActivationFunction:

   @staticmethod
   def step_function_binary(value: float) -> int:
      if value >= 0:
         return 1
      return -1
   
   @staticmethod
   def step_function_bipolar(value: float) -> int:
      if value >= 0:
         return 1
      return -1