import matplotlib.pyplot as plt

depths = [5, 10, 15]
accuracies = [0.725, 0.892, 0.943]

plt.figure()
plt.plot(depths, accuracies, marker='o')
plt.title('Test Accuracy vs Tree Depth')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('accuracy_vs_depth.png')
plt.show()
