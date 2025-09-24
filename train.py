import argparse, os, random, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
def build_model():
    return models.Sequential([
        layers.Input(shape=(32,32,3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32,3,padding='same'), layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(32,3,padding='same'), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(64,3,padding='same'), layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(64,3,padding='same'), layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256), layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze(); y_test = y_test.squeeze()
    model = build_model()
    os.makedirs('artifacts', exist_ok=True)
    ckpt = callbacks.ModelCheckpoint('artifacts/best.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    hist = model.fit(x_train, y_train, validation_split=0.1,
                     epochs=a.epochs, batch_size=a.batch_size,
                     callbacks=[ckpt], verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test accuracy: {test_acc:.4f}')
    model.save('artifacts/model.keras')
    plt.figure(); plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy'])
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(['train','val']); plt.title('Accuracy')
    plt.savefig('artifacts/accuracy.png', dpi=150)
    plt.figure(); plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss'])
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(['train','val']); plt.title('Loss')
    plt.savefig('artifacts/loss.png', dpi=150)
if __name__ == '__main__':
    main()
