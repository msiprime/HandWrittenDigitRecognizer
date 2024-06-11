import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Data for training and validation accuracy and loss
epochs = range(1, 11)
train_accuracy = [0.9123, 0.9855, 0.9911, 0.9940, 0.9958, 0.9965, 0.9965, 0.9983, 0.9980, 0.9980]
train_loss = [0.2931, 0.0469, 0.0286, 0.0202, 0.0133, 0.0113, 0.0098, 0.0054, 0.0063, 0.0052]
val_accuracy = [0.9856, 0.9881, 0.9847, 0.9891, 0.9890, 0.9901, 0.9909, 0.9904, 0.9918, 0.9914]
val_loss = [0.0441, 0.0315, 0.0484, 0.0334, 0.0351, 0.0306, 0.0289, 0.0351, 0.0302, 0.0303]

# Plotting training and validation accuracy
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plot as an image file
image_path = 'training_validation_metrics.png'
plt.savefig(image_path)
plt.close()

# Open the image using an image viewer
img = mpimg.imread(image_path)
imgplot = plt.imshow(img)
plt.show()
