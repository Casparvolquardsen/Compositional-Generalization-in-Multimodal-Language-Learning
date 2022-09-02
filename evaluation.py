import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def get_relative_confusion_matrix(confusion_matrix):
    # confusion_matrix_relative = confusion_matrix / np.sum(confusion_matrix, axis=1)
    confusion_matrix_relative = np.zeros_like(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        confusion_matrix_relative[i] = confusion_matrix[i] * 100 / np.sum(confusion_matrix[i]) if np.sum(
            confusion_matrix[i]) > 0 else 0

    return confusion_matrix_relative


def create_confusion_matrix_plt(plot_matrix, title, save_path, floating):
    dictionary = ["put down", "picked up", "pushed left", "pushed right",
                  "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                  "red", "green", "blue", "yellow", "white", "brown"]
    fig, ax = plt.subplots()
    im = ax.imshow(plot_matrix, vmin=0.0, vmax=np.max(plot_matrix))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(dictionary)), labels=dictionary)
    ax.set_yticks(np.arange(len(dictionary)), labels=dictionary)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(dictionary)):
        for j in range(len(dictionary)):
            my_formatter = "{0:2.2f}" if floating else "{0:4.0f}"
            text = ax.text(j, i, f"{my_formatter.format(plot_matrix[i, j])}{'%' if floating else ''}",
                           ha="center", va="center", color="w" if plot_matrix[i, j] < np.max(plot_matrix) / 2 else "0")

    ax.set_title(title)
    # plt.figtext(0.1, 0.5, 'Tatsächlich', horizontalalignment='center', va="center", rotation=90)
    # plt.figtext(0.5, 0.01, 'Vorhersage', horizontalalignment='center')
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fig.tight_layout()
    fig.set_size_inches(24, 18)
    fig.tight_layout()
    # Path(save_path).mkdir(exist_ok=True)
    # plt.savefig(f"{save_path}{title}.png", dpi=100, bbox_inches='tight')

    return plt


def get_evaluation(model, data_loader, device, description=""):
    dictionary = ["put down", "picked up", "pushed left", "pushed right",
                  "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                  "red", "green", "blue", "yellow", "white", "brown"]
    confusion_matrix = np.zeros((len(dictionary), len(dictionary)))
    model.eval()

    wrong_predictions = []

    with torch.no_grad():
        outputs = torch.zeros((len(data_loader.dataset)), 3, 19)
        labels = torch.zeros((len(data_loader.dataset), 3))
        correct_sentences = 0

        i = 0
        for (frames_batch, joints_batch, label_batch) in tqdm(data_loader, desc=description):
            frames_batch = frames_batch.to(device=device)  # (N, L, c, w, h)
            joints_batch = joints_batch.to(device=device)  # (N, L, j)

            output_batch = model(frames_batch, joints_batch)

            outputs[i:i + data_loader.batch_size] = output_batch.to(torch.device("cpu"))
            labels[i:i + data_loader.batch_size] = label_batch
            i += data_loader.batch_size

        _, action_outputs = torch.max(outputs[:, 0, :], dim=1)
        _, color_outputs = torch.max(outputs[:, 1, :], dim=1)
        _, object_outputs = torch.max(outputs[:, 2, :], dim=1)

        for n in range(outputs.shape[0]):
            confusion_matrix[int(labels[n, 0].item()), (action_outputs[n].item())] += 1
            confusion_matrix[int(labels[n, 1].item()), (color_outputs[n].item())] += 1
            confusion_matrix[int(labels[n, 2].item()), (object_outputs[n].item())] += 1

            action_correct = torch.sum(action_outputs[n] == labels[n, 0])
            color_correct = torch.sum(color_outputs[n] == labels[n, 1])
            object_correct = torch.sum(object_outputs[n] == labels[n, 2])

            if action_correct and color_correct and object_correct:
                correct_sentences += 1

            if (not action_correct) or (not color_correct) or (not object_correct):
                wrong_predictions.append(f"{description} sequence_{n:04d}, "
                                         f"predicted: {dictionary[action_outputs[n].item()]} {dictionary[color_outputs[n].item()]} {dictionary[object_outputs[n].item()]}, "
                                         f"actual:    {dictionary[int(labels[n, 0].item())]} {dictionary[int(labels[n, 1].item())]} {dictionary[int(labels[n, 2].item())]}")

    sentence_wise_accuracy = correct_sentences * 100 / len(data_loader.dataset)

    return confusion_matrix, wrong_predictions, sentence_wise_accuracy
