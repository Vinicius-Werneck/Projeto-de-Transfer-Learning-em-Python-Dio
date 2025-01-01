import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from PIL import Image

# Configuração do dataset
dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
data_dir = tf.keras.utils.get_file(
    "kagglecatsanddogs_5340.zip", origin=dataset_url, extract=True
)
data_dir = os.path.join(os.path.dirname(data_dir), "PetImages")  # Caminho correto
print(f"Arquivo zip salvo em: {data_dir}")
print(f"Dataset extraído para: {os.path.dirname(data_dir)}")


# Função para remover e corrigir imagens inválidas
def clean_and_fix_images(data_dir):
    num_skipped = 0
    for category in ("Cat", "Dog"):
        category_dir = os.path.join(data_dir, category)
        for fname in os.listdir(category_dir):
            fpath = os.path.join(category_dir, fname)
            try:
                # Abre a imagem e garante que seja RGB com dimensões válidas
                with Image.open(fpath) as img:
                    img = img.convert("RGB")  # Converte para RGB
                    img.verify()  # Verifica a integridade
                    # Reescreve a imagem no formato correto
                    img = img.resize((160, 160))
                    img.save(fpath)
            except (IOError, SyntaxError):
                # Remove imagens corrompidas
                num_skipped += 1
                os.remove(fpath)
    print(f"Corrigidas ou removidas {num_skipped} imagens problemáticas.")


# Limpeza e correção das imagens
clean_and_fix_images(data_dir)

# Carregando os datasets de treino e validação
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(160, 160),
    batch_size=32,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(160, 160),
    batch_size=32,
)

# Normalização dos pixels das imagens
normalization_layer = layers.Rescaling(1.0 / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Pré-carregando o modelo MobileNetV2
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3), include_top=False, weights="imagenet"
)
pretrained_model.trainable = False  # Congela os pesos do modelo pré-treinado

# Construção do modelo
model = models.Sequential(
    [
        pretrained_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(2, activation="softmax"),  # Duas classes: gatos e cachorros
    ]
)

# Compilação do modelo
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Treinamento do modelo
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)m  çen


enzo

# Avaliação do modelo
loss, accuracy = model.evaluate(validation_dataset)
print(f"Accuracy: {accuracy * 100:.2f}%")

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
