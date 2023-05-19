import csv
import matplotlib.pyplot as plt

def save_training_data_to_csv(file_path, model_data):
    model_name = model_data["name"]
    with open(file_path / f"{model_name}.csv", 'w', newline='') as csvfile:
        fieldnames = ['train_losses', 'test_losses', 'test_accuracies', 'test_top5_accuracies']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(model_data["train_losses"])):
            writer.writerow({
                'train_losses': model_data["train_losses"][i],
                'test_losses': model_data["test_losses"][i],
                'test_accuracies': model_data["test_accuracies"][i],
                'test_top5_accuracies': model_data["test_top5_accuracies"][i],
            })
            
def plot_multiple_losses_and_accuracies(model_data_list):
    """
    Plots training and testing losses and accuracies for multiple models.

    Args:
        model_data_list (list of dict): A list of dictionaries containing the following keys:
            - 'name' (str): The name of the model (for the legend)
            - 'train_losses' (list): Training losses per epoch
            - 'test_losses' (list): Testing losses per epoch
            - 'test_accuracies' (list): Testing accuracies per epoch
    """
    if not isinstance(model_data_list, list):
        model_data_list = [model_data_list]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for model_data in model_data_list:
        plt.plot(model_data['train_losses'], label=model_data['name'] + ' Train')
        plt.plot(model_data['test_losses'], label=model_data['name'] + ' Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Losses')

    plt.subplot(1, 2, 2)
    for model_data in model_data_list:
        plt.plot(model_data['test_accuracies'], label=model_data['name'])
        plt.plot(model_data['test_top5_accuracies'], label=model_data['name'] + 'top5')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Test Accuracies')

    plt.savefig("results/resnet8.jpg")

    plt.show()
