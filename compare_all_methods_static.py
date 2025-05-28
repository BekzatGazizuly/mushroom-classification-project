import matplotlib.pyplot as plt

criteria = ['Gini', 'Entropy', 'Scaled Entropy', 'Random Forest']
accuracies = [0.9427, 0.9263, 0.9263, 0.9654]

plt.figure(figsize=(8, 5))
bars = plt.bar(criteria, accuracies, color='skyblue')
plt.ylabel('Accuracy')
plt.title('Comparison of Splitting Criteria and Random Forest')
plt.ylim(0.9, 0.97)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add values above bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("comparison_all_methods.png")
plt.show()
