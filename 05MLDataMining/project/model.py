import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC

""" Functions for Assessment 3: ZZSC5836 Data Mining and Machine Learning: """

""" # 2. Develop a dense neural network with one hidden layer.
         Vary the number of hidden neurons to be 5, 10, 15, and 20 in order to investigate the
         performance of the model using Stochastic Gradient Descent (SGD).
         Determine the optimal number of neurons in the hidden layer from the range of values considered."""


def build_and_train_model(X_train, X_test, y_train, y_test, hidden_neurons, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(hidden_neurons, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(4, activation='softmax'))  # 4 classes
    model.compile(optimizer=SGD(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]  # Accuracy
    return accuracy, history


def evaluate_hidden_neurons(df, neuron_options=None):
    if neuron_options is None:
        neuron_options = [5, 10, 15, 20]
    # Split features and target
    X = df.drop("RingAgeClass", axis=1).values
    y = df["RingAgeClass"].astype(int).values
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    scores = {}
    for neurons in neuron_options:
        acc, _ = build_and_train_model(X_train, X_test, y_train, y_test, neurons)
        print(f"Hidden Neurons: {neurons} → Accuracy: {acc:.4f}")
        scores[neurons] = acc
    # Determine optimal configuration
    best_neurons = max(scores, key=scores.get)
    # print(f"\nBest number of hidden neurons: {best_neurons} with Accuracy: {scores[best_neurons]:.4f}")
    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.title("Accuracy vs Hidden Layer Neurons")
    plt.xlabel("Number of Hidden Neurons")
    plt.xticks(neuron_options)
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    save_plot('Accuracy-vs-Hidden-Layer-Neurons')
    plt.show(block=False)
    return best_neurons, scores


""" 3. Investigate the effect of learning rate (using SGD) for the selected dataset
       (using the optimal number of hidden neurons)."""


def evaluate_learning_rates(df, best_neurons, learning_rate_options=None):
    if learning_rate_options is None:
        learning_rate_options = [0.1, 0.01, 0.001]
    # Split features and target
    X = df.drop("RingAgeClass", axis=1).values
    y = df["RingAgeClass"].astype(int).values
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    scores = {}
    for lr in learning_rate_options:
        acc, _ = build_and_train_model(X_train, X_test, y_train, y_test, best_neurons, learning_rate=lr)
        print(f"Learning Rate: {lr} → Accuracy: {acc:.4f}")
        scores[lr] = acc
    # Determine optimal learning rate
    best_lr = max(scores, key=scores.get)
    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o', color='green')
    plt.title(f"Accuracy vs Learning Rate (Hidden Neurons = {best_neurons})")
    plt.xlabel("Learning Rate")
    plt.xscale("log")
    plt.xticks(learning_rate_options)
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    save_plot('Accuracy-vs-Learning-Rate')
    plt.show(block=False)
    return best_lr, scores


""" # 1. Analyse and visualise the given datasets by reporting the distribution of classes,
         distribution of features and any other visualisation you find appropriate."""


def plot_ring_age_correlation_heatmap(df):
    """Draws a correlation heatmap including RingAge and all other features."""
    df_copy = df.copy()
    df_copy['RingAge'] = df_copy['Rings'] + 1.5
    df_copy.drop(columns=['Rings'], inplace=True)
    # Compute correlation matrix
    corr_matrix = df_copy.corr(numeric_only=True)
    # Plot heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdYlGn',
                linewidths=0.5, annot_kws={"size": 8})
    plt.title('Correlation Heatmap', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_plot('Abalone-Features-RingAge-CorrelationHeatmap')
    plt.show(block=False)


def plot_ring_age_percentage_histogram(df, class_col='RingAgeClass'):
    """Draws a histogram showing the percentage of abalones in each RingAgeClass."""
    # Compute percentage distribution
    class_percent = df[class_col].value_counts(normalize=True).sort_index() * 100
    # Plot
    plt.figure(figsize=(6, 5))
    bars = plt.bar(class_percent.index.astype(str), class_percent.values, color='salmon', edgecolor='black')
    # Annotate bars with percentages
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.1f}%', ha='center', fontsize=10)
    plt.xticks(ticks=[0, 1, 2, 3], labels=['Class 1 (≤7)', 'Class 2 (7–10)', 'Class 3 (10–15)', 'Class 4 (>15)'])
    plt.xlabel('Ring Age Class')
    plt.ylabel('Percentage of Abalones')
    plt.title('Percentage Distribution of Abalones by RingAgeClass')
    plt.ylim(0, max(class_percent.values) + 10)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_plot('Abalone-AgeClass-Histogram')
    plt.show(block=False)


def clean_data(df_abalone):
    # One Hot Encoding for Sex Column
    df_cleaned = pd.get_dummies(df_abalone, columns=['Sex'])
    # Convert boolean Sex_* columns to integers (0/1)
    sex_cols = [col for col in df_cleaned.columns if col.startswith('Sex_')]
    df_cleaned[sex_cols] = df_cleaned[sex_cols].astype(int)
    # print(f'\n# Clean abalone data (e.g. One Hot Encoding for Sex Column).\n{df_cleaned.head(5)}')
    return df_cleaned


def classify_ring_age(age):
    # Classify Ring-Age
    if age <= 7:
        return 0
    elif age <= 10:
        return 1
    elif age <= 15:
        return 2
    else:
        return 3


def age_classify(df_abalone):
    # Add 1.5 to 'Rings' and rename the column to 'RingAge'
    df_abalone['RingAge'] = df_abalone['Rings'] + 1.5
    # Classify Ring Age:
    df_abalone['RingAgeClass'] = df_abalone['RingAge'].apply(classify_ring_age)
    # Drop column RingAge
    df_abalone.drop(columns=['Rings', 'RingAge'], inplace=True)
    return df_abalone


""" Helper functions: Not part of Assessment 2 """


def config_options():
    """ Set options and configurations """
    # set wider console display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    # Explicitly convert string to an integer
    pd.set_option('future.no_silent_downcasting', True)


def load_data(file_path, file_name, column_names):
    # Load the dataset
    return pd.read_csv(file_path + file_name, names=column_names)


def save_plot(filename):
    save_dir = r"./plots"  # Output directory to generate plots/ .png files
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
    save_path = os.path.join(save_dir, filename+'.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the plot
    # print(f"Plot saved to: {save_path}")


def main():
    """Main function to execute the Linear Model Data Analysis."""
    # set options and configurations:
    config_options()
    # load abalone data file
    file_path = './data/'
    file_name = 'abalone.data'
    column_names = ["Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight",
                    "ShellWeight", "Rings"]
    df_abalone_data = load_data(file_path, file_name, column_names)
    # print(f'\nRaw Abalone Data.\n{df_abalone_data.head()}')
    # Clean abalone data: OneHotEncoding for Sex
    df_clean_abalone = clean_data(df_abalone_data)
    """ # 1. Analyse and visualise the given datasets by reporting the distribution of classes,
    distribution of features and any other visualisation you find appropriate."""
    # print(f'\nClean Abalone Data (One Hot Encoded Sex).\n{df_clean_abalone.head(5)}')
    plot_ring_age_correlation_heatmap(df_clean_abalone)
    # Classify abalone data:
    df_abalone = age_classify(df_clean_abalone)
    # print(f'\nAge Classified Clean Abalone Data: df_abalone.head()\n{df_abalone.head(5)}')
    plot_ring_age_percentage_histogram(df_abalone)
    # print("Current working directory:", os.getcwd())
    # df_abalone = df_abalone["RingAgeClass"].astype(int) - 1
    print(f'\nTraining Neural Network with df_abalone.head()\n{df_abalone.head(5)}'
          f'\nType of columns in df_abalone: {df_abalone.dtypes}'
          f'\nNumber of rows in df_abalone: {df_abalone.shape[0]}')
    neuron_options = [5, 10, 15, 20]
    best_neurons, scores = evaluate_hidden_neurons(df_abalone, neuron_options)
    print(f"\nBest number of hidden neurons: {best_neurons} with Accuracy: {scores[best_neurons]:.4f}")
    learning_rate_options = [0.1, 0.01, 0.001]
    best_learning_rate, scores = evaluate_learning_rates(df_abalone, best_neurons, learning_rate_options)
    print(f"\nBest Learning Rate: {best_learning_rate} for Best hidden neurons: {best_neurons}"
          f"is with Accuracy: {scores[best_learning_rate]:.4f}")


if __name__ == "__main__":
    main()
