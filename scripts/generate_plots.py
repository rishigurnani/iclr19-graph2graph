import os
import numpy as np
import matplotlib.pyplot as plt

def create_data(directory):
    success = []
    diversity = []
    # Read success and diversity data from given directory
    for filename in os.listdir(directory):
        if filename.startswith("analyze.") and not filename.endswith(".swp"):
            with open(filename, 'r') as file:
                lines = file.readlines()
                dot = filename.index('.') + 1
                epoch = int(filename[dot:])
                for line in lines:
                    words = line.split()
                    if "success" in line:
                        success.append((epoch, float(words[2])))
                    if "diversity" in line:
                        diversity.append((epoch, np.nan_to_num(float(words[2]))))
    # Sort data based on epoch #
    success = sorted(success, key = lambda x: x[0])
    diversity = sorted(diversity, key = lambda x: x[0])
    return success, diversity

if __name__ == "__main__":
    directory = './'
    success, diversity = create_data(directory)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(*zip(*success))
    ax1.set_title('Success')
    ax1.set_xticks(np.arange(0, 20, 2))
    ax1.set_ylabel('Success rate')
    ax1.set_xlabel('Epoch #')
    ax2 = fig.add_subplot(212)
    ax2.plot(*zip(*diversity))
    ax2.set_title('Diversity')
    ax2.set_xticks(np.arange(0, 20, 2))
    ax2.set_ylabel('Diversity score')
    ax2.set_xlabel('Epoch #')
    fig.tight_layout()
    plt.savefig('analysis.png')