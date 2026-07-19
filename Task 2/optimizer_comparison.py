import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

def create_model():
    """Creates a simple feedforward neural network."""
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

def main():
    # 1. Load a small dataset (MNIST)
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 3. Optimizers to test
    optimizers = {
        'SGD': tf.keras.optimizers.SGD(),
        'RMSProp': tf.keras.optimizers.RMSprop(),
        'Adam': tf.keras.optimizers.Adam()
    }

    histories = {}
    epochs = 5  # Keeping epochs small for quick execution

    # Train model separately for each optimizer
    for opt_name, optimizer in optimizers.items():
        print(f"\n--- Training with {opt_name} Optimizer ---")
        
        # 2. Build the neural network using Sequential model
        model = create_model()
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # 4. Record training accuracy and loss
        history = model.fit(
            x_train, y_train, 
            epochs=epochs, 
            validation_data=(x_test, y_test)
        )
        histories[opt_name] = history.history

    # 5. Plot the training loss and accuracy graphs
    plt.figure(figsize=(14, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    for opt_name in optimizers.keys():
        plt.plot(histories[opt_name]['accuracy'], label=f'{opt_name} Train Accuracy')
        plt.plot(histories[opt_name]['val_accuracy'], label=f'{opt_name} Val Accuracy', linestyle='--')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    for opt_name in optimizers.keys():
        plt.plot(histories[opt_name]['loss'], label=f'{opt_name} Train Loss')
        plt.plot(histories[opt_name]['val_loss'], label=f'{opt_name} Val Loss', linestyle='--')
    plt.title('Model Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Save the plot for reference
    plt.savefig('optimizer_comparison.png')
    print("\nTraining complete. Performance graphs saved as 'optimizer_comparison.png'.")
    
    # Show the plots
    plt.show()

    # 6. Compare performance and print simple conclusion
    print("\n--- Final Validation Accuracies ---")
    for opt_name in optimizers.keys():
        final_val_acc = histories[opt_name]['val_accuracy'][-1]
        print(f"{opt_name}: {final_val_acc:.4f}")
    
    best_opt = max(histories.keys(), key=lambda o: histories[o]['val_accuracy'][-1])
    print(f"\nConclusion: Based on validation accuracy after {epochs} epochs, the '{best_opt}' optimizer gave the best results.")

if __name__ == "__main__":
    main()
