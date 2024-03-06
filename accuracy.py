import torch
import numpy as np
from preprocessing import rellenar_con_ceros

# ESTA FUNCION ESTA EN ESTADO DE PRUEBA
def calcular_accuracy(predictions, labels):
    # Ajustar las dimensiones de las predicciones
    predictions = predictions.view(-1, 10)
    probabilities = torch.nn.functional.softmax(predictions, dim=1)

    # Obtener las clases predichas
    _, predicted_classes = torch.max(probabilities, 1)

    # Ajustar las dimensiones para tener una lista de predicciones para cada muestra en el batch
    predicted_classes = predicted_classes.view(-1, 6)

    # Crear una lista de predicciones completas para cada muestra en el batch
    predicted_numbers = []
    for pred in predicted_classes:
        pred = [str(num.item()) for num in pred]
        pred = rellenar_con_ceros(int(''.join(pred)), 6)
        predicted_numbers.append(pred)

    # Comparar las predicciones completas con las etiquetas del batch
    correct = np.array([pred == label for pred, label in zip(predicted_numbers, labels)]).sum().item()

    total = len(labels)

    return correct / total