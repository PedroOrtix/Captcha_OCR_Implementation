import torch
import numpy as np
from preprocessing import rellenar_con_ceros

def evaluate_model(model, dataloader):
    model.eval()
    
    # Initialize variables to store the total correct predictions and total number of images
    total_correct = 0
    total_images = 0
    df = {"label": [], "predicted": []}

    with torch.no_grad():  # this is optional but good practice for memory efficiency
        # Iterate over the validation dataloader
        for batch in dataloader:
            images, _, labels = batch
            for image, label in zip(images, labels):
                # Perform a forward pass through the model
                predictions = model(image.unsqueeze(dim=0))
                # Split the vector per 10 elements
                predictions = predictions.view(-1, 10)
                probabilities = torch.nn.functional.softmax(predictions, dim=1)
            
                # Make an argmax to the probabilities
                _, predicted_classes = torch.max(probabilities, 1)
            
                # Get the ground truth labels
                ground_truth = label

                # Convert the predicted classes to integer tensor and join the tenso in a single number
                predicted_classes = predicted_classes.view(-1, 6).squeeze()
                str_list = [str(num.item()) for num in predicted_classes]
                result = rellenar_con_ceros(int(''.join(str_list)), 6)
            
                # Compare the predicted classes with the ground truth labels and update total correct predictions if they match
                total_correct += np.array(result == ground_truth).sum().item()
                
                df["label"].append(ground_truth)
                df["predicted"].append(result)

            total_images += len(labels)
    
    # Calculate the accuracy
    accuracy = total_correct / total_images

    print(f"Accuracy: {accuracy * 100}%")
    return df, accuracy
