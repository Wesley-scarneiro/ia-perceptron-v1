from configs.parameters import Parameters
from configs.learning_data import LearningData
from learn import Learn
import logging
from datetime import datetime

def main():
    try:
        # Parâmetros de inicialização do treinamento
        Parameters.learning_rate = 0.5
        Parameters.bias_value = 1
        
        # Dados de treinamento e de teste
        data = [
            LearningData(
            [1, 1], 1),
            LearningData([1, 0], -1),
            LearningData([0, 1], -1),
            LearningData([0, 0], -1)
        ]

        # Iniciando a aprendizagem do perceptron
        learn = Learn(data)
        learn.start_learning()
        learn.test_learning_perceptron()

    except Exception as ex:
        logging.error(ex)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename='output.log')
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"{now} - Running perceptron")
    main()
    logging.info(f"{now} - finished perceptron\n")
