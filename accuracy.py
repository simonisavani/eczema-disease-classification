from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Load Trained Model ===
model = load_model("eczema_vs_noneczema_model.keras")

# === Prepare Test Dataset ===
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'datasets/test',  # Make sure this folder exists!
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# === Evaluate Model ===
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

# === Generate Predictions ===
y_pred = model.predict(test_generator)
y_pred_labels = (y_pred > 0.5).astype(int).flatten()
y_true = test_generator.classes

# === Classification Report ===
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_labels, target_names=test_generator.class_indices.keys()))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_generator.class_indices, 
            yticklabels=test_generator.class_indices)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
