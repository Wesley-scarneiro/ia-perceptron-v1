from configs.parameters import Parameters
from configs.learning_data import LearningData
from learn import Learn
import logging
from datetime import datetime

def log_parameters() -> None:
    string = f'''-- Parameters -- 
\tLearning_rate = {Parameters.learning_rate}
\tBias_value = {Parameters.bias_value}
\tInputs_total = {Parameters.inputs_total}'''
    logging.info(string)

def main():
    try:
        # Parâmetros de inicialização do treinamento
        Parameters.learning_rate = 0.3
        Parameters.bias_value = 1
        log_parameters()
        
        # Dados de treinamento e de teste
        data = [
            LearningData([1, 1], 1),
            LearningData([1, 0], -1),
            LearningData([0, 1], -1),
            LearningData([0, 0], -1)
        ]

        # Iniciando a aprendizagem do perceptron
        learn = Learn(data)
        learn.start_learning()
        learn.learning_log()
        learn.test_learning_perceptron()

    except Exception as ex:
        logging.error(ex)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename='output.log')
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-- Running perceptron --")
    logging.info(f"{now} - Running perceptron")
    main()
    print("-- Finished perceptron --")
    logging.info(f"{now} - Finished perceptron\n")
