import os

import matplotlib.pyplot as plt

from .plot_confusion_matrix import plot_confusion_matrix



class Visualizer:
    def __init__(self, output_dir='../plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_curves(self, train_losses, train_accuracies, test_accuracies):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].plot(train_losses)
        axs[0].set_title('Training Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')

        axs[1].plot(train_accuracies, label='Train')
        axs[1].plot(test_accuracies, label='Test')
        axs[1].set_title('Accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plot_file = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(plot_file)
        plt.close()


    def plot_accuracy_per_class(self, epochs, test_accuracies_per_class, class_labels):
        num_classes = len(class_labels)
        for i in range(num_classes):
            plt.plot(epochs, [x[i] for x in test_accuracies_per_class], label=f'Class {class_labels[i]}')

        plt.title('Accuracy per Class every 10 Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plot_file = os.path.join(self.output_dir, 'accuracy_per_class.png')
        plt.savefig(plot_file)
        plt.close()

    def plot_conf_matrix(self, test_Y, test_Y_predict, class_labels):
        # Plot accuracy
        axis, cf = plot_confusion_matrix(
            test_Y, test_Y_predict, class_labels, normalize=False, size=(12, 8))
        plot_file = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(plot_file)
        plt.close()
