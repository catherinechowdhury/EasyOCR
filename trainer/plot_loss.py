import re
import matplotlib.pyplot as plt

iterations = []
losses = []

with open("saved_models/debug_run/log_train.txt", "r", encoding="utf-8") as f:
    for line in f:
        match = re.search(r"\[(\d+)/\d+\] Train loss: ([0-9.]+)", line)

        if match:
            iteration = int(match.group(1))
            loss = float(match.group(2))

            iterations.append(iteration)
            losses.append(loss)

plt.plot(iterations, losses)

plt.xlabel("Iteration")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve")

plt.show()