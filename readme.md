# Perceptron-And

Programa que simula o funcionamento de um perceptron simples que é capaz de resolver o problema do AND, por meio de um treinamento supervisionado.

## Execução

Para iniciar o programa basta executar o arquivo **main.py**

Os dados de execução do programa são registrados no arquivo **output.log**

## Problema do AND

Dado duas entradas binárias X e Y, o perceptron deve ser capaz de resolver corretamente a sentença X ^ Y.
Se a sentença X ^ Y é verdadeira a saída do perceptron deve ser 1, caso ao contrário -1.

Para que o perceptron seja capaz de resolver o problema do AND, é necessário encontrar e ajustar um conjunto de pesos para que seja possível produzir uma saída desejada para cada exemplo de entrada.

### Entradas e saídas esperadas

* 1, 1 ->  1  
* 1, 0 -> -1
* 0, 1 -> -1
* 0, 0 ->  0

## Classes

* **Perceptron**: classe que encapsula os atributos e operações realizadas por um perceptron
    * Contém um bias, neurônios sensoriais (entradas), função de entrada, função de ativação
* **SensoryNeuron**: classe que representa os neurônios sensoriais ou as interfaces de entradas do perceptron
    * Contém um valor de entrada e um peso
* **Parameters**: classe estática que encapsula os parâmetros da taxa de aprendizagem, valor do bias e a quantidade de entradas do perceptron
* **LearnData**: classe que encapsula os dados de treinamento do perceptron
    * Contém uma lista com dois valores de entrada e um valor de saída esperada para essa entrada (target)
* **Learn**: classe que encpsula o algoritmo de treinamento do perceptron
    * Inicializa o perceptron e inicia o treinamento

## Algoritmo de treinamento

O treinamento do perceptron é realizado pela classe Learn que implementa o seguinte algoritmo:

    1 while(True):
    2     updated_weights = False
    3     for data in self.__data:
    4         output = self.__perceptron.output(data.inputs)
    5         if output != data.target:
    6             self.__perceptron.update_synaptic_weights(data.target)
    7             updated_weights = True
    8             break
    9     if (not updated_weights):
    10         break

Antes do início do treinamento, na instanciação do perceptron, é criada uma lista de neurônios sensoriais (que representam a interface de entrada do perceptron) com um bias pré-definido e dois neurônios sensoriais com pesos aleatórios. Após isso, o perceptron é treinado com uma coleção de entradas até que seu conjunto de pesos seja capaz de classificar, pela função de ativação, todas as entradas corretamente.

O **break** na linha **8** é responsável por parar a iteração do **for** na linha **3**. Essa parada é necessária, pois os pesos do perceptron foram atualizados e todas as entradas anteriores deverão ser reprocessadas - para garantir que o perceptron seja capaz de classificá-las corretamente com os novos pesos.

O **break** na linha **10** é o responsável por parar a iteração do **while** na linha **1** e finalizar o treinamento. Se a variável **updated_weights** é **false** na linha **9**, então o perceptron iterou sobre toda a coleção de dados sem a necessidade de alterar os seus pesos e portanto, classificou todas as entradas corretamente.

### Função de entrada e de ativação

A **função de entrada** realiza a somatória do produto entre o valor de entrada e o seu peso. Essa operação é realizada pelo método **_input_function()** do perceptron:

    def _input_function(self) -> float:
            value = 0
            for sensory_neuron in self.__sensory_neurons:
                value += sensory_neuron.value * sensory_neuron.weight
            return value

A saída da função de entrada é direcionada para a **função de ativação**, pertencente a classe estática ActivationFunction, que utiliza uma função step binária com decisão em 0:

    @staticmethod
    def step_function_binary(value: float) -> int:
        if value >= 0:
            return 1
        return -1

## Output

Exemplo de uma saída do perceptron com poucas iterações de treinamento:

    2024-04-07 01:07:15 - Running perceptron
    -- Parameters -- 
        Learning_rate = 0.3
        Bias_value = 1
        Inputs_total = 2
    -- Creating percepton --
        Perceptron=[Id=1, SensoryNeurons=[[Id=1, Value=1, Weight=0.139], [Id=2, Value=None, Weight=0.781], [Id=3, Value=None, Weight=0.934]]]
    -- Starting learning --
        -- Iterating data --
            - LearningData=[Inputs=[1, 1], Target=1]
            - output=1
            - LearningData=[Inputs=[1, 0], Target=-1]
            - output=1
            -- Update weights --
                - SensoryNeurons=[[Id=1, Value=1, Weight=-0.161], [Id=2, Value=1, Weight=0.481], [Id=3, Value=0, Weight=0.934]]
        -- Finished iteration --
        -- Iterating data --
            - LearningData=[Inputs=[1, 1], Target=1]
            - output=1
            - LearningData=[Inputs=[1, 0], Target=-1]
            - output=1
            -- Update weights --
                - SensoryNeurons=[[Id=1, Value=1, Weight=-0.461], [Id=2, Value=1, Weight=0.181], [Id=3, Value=0, Weight=0.934]]
        -- Finished iteration --
        -- Iterating data --
            - LearningData=[Inputs=[1, 1], Target=1]
            - output=1
            - LearningData=[Inputs=[1, 0], Target=-1]
            - output=-1
            - LearningData=[Inputs=[0, 1], Target=-1]
            - output=1
            -- Update weights --
                - SensoryNeurons=[[Id=1, Value=1, Weight=-0.761], [Id=2, Value=0, Weight=0.181], [Id=3, Value=1, Weight=0.634]]
        -- Finished iteration --
        -- Iterating data --
            - LearningData=[Inputs=[1, 1], Target=1]
            - output=1
            - LearningData=[Inputs=[1, 0], Target=-1]
            - output=-1
            - LearningData=[Inputs=[0, 1], Target=-1]
            - output=-1
            - LearningData=[Inputs=[0, 0], Target=-1]
            - output=-1
        -- Finished iteration --
    -- Finished learning --
    -- Learning log --
        - Perceptron=[Id=1, SensoryNeurons=[[Id=1, Value=1, Weight=-0.761], [Id=2, Value=0, Weight=0.181], [Id=3, Value=0, Weight=0.634]]]
        - Iterations total=4
    -- Test learning perceptron --
    Input = [1, 1] | Output = 1 | Correct? True
    Input = [1, 0] | Output = -1 | Correct? True
    Input = [0, 1] | Output = -1 | Correct? True
    Input = [0, 0] | Output = -1 | Correct? True
    2024-04-07 01:07:15 - Finished perceptron