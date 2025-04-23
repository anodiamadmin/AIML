import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop, Adagrad
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.utils import to_categorical


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
    plots_dir = os.path.join(os.getcwd(), 'plots')  # Absolute path
    os.makedirs(plots_dir, exist_ok=True)  # Create if not exists
    full_path = os.path.join(plots_dir, f"{filename}.png")
    plt.savefig(full_path)
    print(f"\nPlot saved to: {full_path}")


""" Functions for Assessment 3: ZZSC5836 Data Mining and Machine Learning: """


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
    save_plot('1_Abalone-Features-RingAge-CorrelationHeatmap')
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
    save_plot('1_Abalone-AgeClass-Histogram')
    plt.show(block=False)


def clean_data(df_abalone):
    # One Hot Encoding for Sex Column
    df_cleaned = pd.get_dummies(df_abalone, columns=['Sex'])
    # Convert boolean Sex_* columns to integers (0/1)
    sex_cols = [col for col in df_cleaned.columns if col.startswith('Sex_')]
    df_cleaned[sex_cols] = df_cleaned[sex_cols].astype(int)
    # print(f'\n# Clean abalone data (e.g., One Hot Encoding for Sex Column).\n{df_cleaned.head(5)}')
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


def setup_load_wrangle_data():
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
    return df_clean_abalone


""" # 1. Analyse and visualise the given datasets by reporting the distribution of classes,
         distribution of features and any other visualisation you find appropriate."""


def exploratory_analysis(df_clean_abalone):
    print(f'\n--------------------------------------------------------------------------')
    print(f'1.\tAnalyse and visualise the given datasets by reporting the distribution of classes, '
          f'\n\tdistribution of features and any other visualisation you find appropriate.')
    print(f'--------------------------------------------------------------------------')
    # print(f'\nClean Abalone Data (One Hot Encoded Sex).\n{df_clean_abalone.head(5)}')
    plot_ring_age_correlation_heatmap(df_clean_abalone)
    # Classify abalone data:
    df_abalone = age_classify(df_clean_abalone)
    # print(f'\nAge Classified Clean Abalone Data: df_abalone.head()\n{df_abalone.head(5)}')
    plot_ring_age_percentage_histogram(df_abalone)
    print(f'Cleaned Data: df_abalone.head()\n{df_abalone.head(5)}'
          f'\n--------------------------------------------------------------------------'
          f'\nType of columns in df_abalone: {df_abalone.dtypes}'
          f'\n--------------------------------------------------------------------------'
          f'\nNumber of rows in df_abalone: {df_abalone.shape[0]}'
          f'\n--------------------------------------------------------------------------')
    output_classes = np.unique(df_abalone["RingAgeClass"])
    print(f'Output classes: {output_classes}')
    print(f'--------------------------------------------------------------------------')
    return df_abalone


""" # 2. Develop a dense neural network with one hidden layer.
         Vary the number of hidden neurons to be 5, 10, 15, 20 in order to investigate the
         performance of the model using Stochastic Gradient Descent (SGD).
         Determine the optimal number of neurons in the hidden layer from the range of values considered."""


def split_data_for_train_test(df, test_size, random_state):
    # Split features and target
    X = df.drop("RingAgeClass", axis=1).values
    y = df["RingAgeClass"].astype(int).values
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size,
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def build_train_evaluate_classification_model(df, test_size=0.4, random_state=42,
                                              hidden_neurons=5, learning_rate=0.01, hidden_layers=1,
                                              optimizer_name="SGD",
                                              non_softmax_activation='relu',
                                              loss_function='sparse_categorical_crossentropy',
                                              softmax_classes=4, epochs=100, batch_size=32, verbose=0):
    # Fixed train/test split
    X_train, X_test, y_train, y_test = split_data_for_train_test(df, test_size=test_size,
                                                                         random_state=random_state)
    # Optimizer selection dictionary
    optimizers = {
        "SGD": SGD(learning_rate=learning_rate),
        "Adam": Adam(learning_rate=learning_rate),
        "Nadam": Nadam(learning_rate=learning_rate),
        "RMSprop": RMSprop(learning_rate=learning_rate),
        "Adagrad": Adagrad(learning_rate=learning_rate)
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    # Set up model fresh for each run
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, activation=non_softmax_activation))
    model.add(Dense(softmax_classes, activation='softmax'))
    optimizer = optimizers[optimizer_name]
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose,
              validation_data=(X_test, y_test))
    acc = model.evaluate(X_test, y_test, verbose=verbose)[1]
    return acc


def evaluate_model_with_confidence_interval(df, n_runs, ci, **model_kwargs):
    accuracies = []
    for i in range(n_runs):
        print(f"Run {i + 1}/{n_runs}...")
        acc = build_train_evaluate_classification_model(df, **model_kwargs)
        accuracies.append(acc)
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_err = st.sem(accuracies)
    confidence_interval = st.t.interval(ci, len(accuracies) - 1, loc=mean_acc, scale=std_err)
    return mean_acc, confidence_interval


def evaluate_neuron_options(df, optimizer_name="SGD", hidden_neurons_list=None,
                            n_runs=10, ci=0.95, test_size=0.4):
    if hidden_neurons_list is None:
        hidden_neurons_list = [5, 10, 15, 20]
    mean_accuracies = []
    confidence_intervals = []
    best_neuron = None
    best_accuracy = 0.0
    for hidden_neurons in hidden_neurons_list:
        print(f"\nEvaluating model with {hidden_neurons} hidden neurons...")
        acc_list = []
        for run in range(n_runs):
            random_state = 42 + run  # different seed for each run
            acc = build_train_evaluate_classification_model(
                df,
                optimizer_name=optimizer_name,
                hidden_neurons=hidden_neurons,
                random_state=random_state,
                test_size=test_size
            )
            acc_list.append(acc)
        acc_array = np.array(acc_list)
        mean_acc = np.mean(acc_array)
        std_err = st.sem(acc_array)
        ci_bounds = st.t.interval(ci, len(acc_array) - 1, loc=mean_acc, scale=std_err)
        mean_accuracies.append(mean_acc)
        confidence_intervals.append((mean_acc - ci_bounds[0], ci_bounds[1] - mean_acc))
        # print(f"Hidden Neurons: {hidden_neurons}")
        print(f"  Mean Accuracy: {mean_acc:.5f}")
        print(f"  95% Confidence Interval: ({ci_bounds[0]:.5f}, {ci_bounds[1]:.5f})")
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_neuron = hidden_neurons
    # Plotting
    plt.figure(figsize=(6, 6))
    lower_bounds, upper_bounds = zip(*confidence_intervals)
    plt.errorbar(hidden_neurons_list, mean_accuracies, yerr=[lower_bounds, upper_bounds],
                 fmt='-o', capsize=5, color='blue', ecolor='red')
    plt.title(f"{optimizer_name}: Mean Accuracy vs Hidden Layer Neurons")
    plt.xlabel("Number of Hidden Neurons")
    plt.ylabel("Mean Accuracy")
    plt.grid(True)
    plt.xticks(hidden_neurons_list)
    # plt.ylim(0, 1)
    print(f"\nBest option: {best_neuron} hidden neurons with accuracy {best_accuracy:.5f}")
    save_plot(f'2_{optimizer_name}-Accuracy-vs-Hidden-Layer-Neurons')
    plt.show(block=False)
    print(f'--------------------------------------------------------------------------')
    return best_neuron


""" # 3. Investigate the effect of learning rate (using SGD) for the selected dataset
         (using the optimal number of hidden neurons)."""


def evaluate_learning_rate(df, best_neuron_option, optimizer_name="SGD" ,learning_rate_list=None,
                           n_runs=5, ci=0.95, test_size=0.4):
    if learning_rate_list is None:
        learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
    mean_accuracies = []
    confidence_intervals = []
    best_lr = None
    best_accuracy = 0.0
    for lr in learning_rate_list:
        print(f"\nEvaluating model with {best_neuron_option} hidden neurons & learning rate = {lr}...")
        acc_list = []
        for run in range(n_runs):
            random_state = 100 + run  # Different random seed per run
            acc = build_train_evaluate_classification_model(
                df,
                optimizer_name=optimizer_name,
                hidden_neurons=best_neuron_option,
                learning_rate=lr,
                random_state=random_state,
                test_size=test_size
            )
            acc_list.append(acc)
        acc_array = np.array(acc_list)
        mean_acc = np.mean(acc_array)
        std_err = st.sem(acc_array)
        ci_bounds = st.t.interval(ci, len(acc_array) - 1, loc=mean_acc, scale=std_err)
        mean_accuracies.append(mean_acc)
        confidence_intervals.append((mean_acc - ci_bounds[0], ci_bounds[1] - mean_acc))
        print(f"  Mean Accuracy: {mean_acc:.5f}")
        print(f"  95% Confidence Interval: ({ci_bounds[0]:.5f}, {ci_bounds[1]:.5f})")
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_lr = lr
    # Plotting
    plt.figure(figsize=(6, 6))
    lower_bounds, upper_bounds = zip(*confidence_intervals)
    plt.errorbar(learning_rate_list, mean_accuracies, yerr=[lower_bounds, upper_bounds],
                 fmt='-o', capsize=5, color='green', ecolor='orange')
    plt.title(f"{optimizer_name}: Mean Accuracy vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.xscale('log')
    plt.ylabel("Mean Accuracy")
    plt.grid(True)
    plt.xticks(learning_rate_list)
    plt.ylim(0.65, 0.77)  # Adjust this range based on your actual accuracy values
    print(f"\nBest option: Learning rate = {best_lr} with accuracy {best_accuracy:.5f}")
    save_plot(f'3_{optimizer_name}Accuracy-vs-Learning-Rate')
    plt.show(block=False)
    print(f'--------------------------------------------------------------------------')
    return best_lr


""" # 4. Investigate the effect on a different number of hidden layers:  
         Now modify the model by adding another hidden layer. Use the optimal number of hidden neurons
         from Step 2 for both the layers and the optimal learning rate from Step 3.
         Investigate the effect of this change in the number of hidden layers (using SGD). """


def evaluate_hidden_layers(df, best_learning_rate, optimizer_name="SGD", hidden_layers_list=None,
                           n_runs=5, ci=0.95, test_size=0.4):
    if hidden_layers_list is None:
        hidden_layers_list = [1, 2, 3]
    mean_accuracies = []
    confidence_intervals = []
    best_hidden_layers = None
    best_accuracy = 0.0
    for layers in hidden_layers_list:
        print(f"\nEvaluating model with learning_rate={best_learning_rate} & {layers} hidden layer(s)...")
        acc_list = []
        for run in range(n_runs):
            random_state = 200 + run  # Different random seed per run
            acc = build_train_evaluate_classification_model(
                df,
                optimizer_name=optimizer_name,
                learning_rate=best_learning_rate,
                hidden_layers=layers,
                random_state=random_state,
                test_size=test_size
            )
            acc_list.append(acc)
        acc_array = np.array(acc_list)
        mean_acc = np.mean(acc_array)
        std_err = st.sem(acc_array)
        ci_bounds = st.t.interval(ci, len(acc_array) - 1, loc=mean_acc, scale=std_err)
        mean_accuracies.append(mean_acc)
        confidence_intervals.append((mean_acc - ci_bounds[0], ci_bounds[1] - mean_acc))
        print(f"  Mean Accuracy: {mean_acc:.5f}")
        print(f"  95% Confidence Interval: ({ci_bounds[0]:.5f}, {ci_bounds[1]:.5f})")
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_hidden_layers = layers
    # Plotting
    plt.figure(figsize=(6, 6))
    lower_bounds, upper_bounds = zip(*confidence_intervals)
    plt.errorbar(hidden_layers_list, mean_accuracies, yerr=[lower_bounds, upper_bounds],
                 fmt='-o', capsize=5, color='purple', ecolor='brown')
    plt.title(f"{optimizer_name}: Mean Accuracy vs Number of Hidden Layers")
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Mean Accuracy")
    plt.grid(True)
    plt.xticks(hidden_layers_list)
    plt.ylim(0.65, 0.77)  # Adjust based on your actual values
    print(f"\nBest option: {best_hidden_layers} hidden layer(s) with accuracy {best_accuracy:.5f}")
    save_plot(f'4_{optimizer_name}-Accuracy-vs-Hidden-Layers')
    plt.show(block=False)
    print(f'--------------------------------------------------------------------------')
    return best_hidden_layers, best_accuracy


""" # 6. Take the final optimal model among all the above cases and show the confusion matrix
         and ROC/AUC curve for different classes of the multi-class problem. """

def plot_confusion_matrix_and_roc(df, best_optimizer, best_number_of_layers, best_learning_rate, best_neurons):
    # Prepare data
    X = df.drop("RingAgeClass", axis=1).values
    y = df["RingAgeClass"].astype(int).values
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    for _ in range(best_number_of_layers):
        model.add(Dense(best_neurons, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 output classes
    # Choose optimizer
    if best_optimizer.upper() == 'ADAM':
        optimizer = Adam(learning_rate=best_learning_rate)
    else:
        optimizer = SGD(learning_rate=best_learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, validation_data=(X_test, y_test))
    # Predict labels and probabilities
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.grid(False)
    save_plot(f'6_{best_optimizer}-Confusion_Matrix')
    plt.show(block=False)
    # --- ROC Curve & AUC ---
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot ROC curves
    plt.figure(figsize=(7, 6))
    colors = ['blue', 'green', 'red', 'purple']
    for i in range(4):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})", color=colors[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Multi-Class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    save_plot(f'6_{best_optimizer}-ROC_Curve')
    plt.show(block=False)


def main():
    df_clean_abalone = setup_load_wrangle_data()
    """ Exploratory Data Analysis."""
    df_abalone = exploratory_analysis(df_clean_abalone)
    """ SGD """
    print(f'\n\nEvaluating SGD Optimizer')
    print(f'--------------------------------------------------------------------------')
    """ Vary the number of hidden neurons to investigate the performance of the model using (SGD) """
    hidden_neuron_list = [5, 10, 15, 20]
    n_runs = 10
    sgd_best_neuron_option = evaluate_neuron_options(df_abalone, n_runs=n_runs, hidden_neurons_list=hidden_neuron_list)
    """ Investigate the effect of learning rate (using SGD) """
    learning_rate_list = [0.1, 0.01, 0.001, 0.0001]
    sgd_best_learning_rate = evaluate_learning_rate(df_abalone, sgd_best_neuron_option, learning_rate_list=learning_rate_list)
    """ Investigate the effect of a varying number of hidden layers """
    hidden_layers_list =  [1, 2, 3]
    sgd_best_hidden_layers, sgd_best_accuracy = evaluate_hidden_layers(df_abalone, sgd_best_learning_rate,  hidden_layers_list=hidden_layers_list)
    """ Adam """
    print(f'\n\nEvaluating Adam Optimizer')
    print(f'--------------------------------------------------------------------------')
    """ Vary the number of hidden neurons to investigate the performance of the model using (Adam) """
    adam_best_neuron_option = evaluate_neuron_options(df_abalone, optimizer_name="Adam",
                                                 n_runs=n_runs, hidden_neurons_list=hidden_neuron_list)
    """ Investigate the effect of learning rate (using Adam) """
    adam_best_learning_rate = evaluate_learning_rate(df_abalone, adam_best_neuron_option, optimizer_name="Adam",
                                                learning_rate_list=learning_rate_list)
    """ Investigate the effect of a varying number of hidden layers """
    adam_best_hidden_layers, adam_best_accuracy = evaluate_hidden_layers(df_abalone, adam_best_learning_rate, optimizer_name="Adam",
                           hidden_layers_list=hidden_layers_list)
    """ # 5. Investigate the effect of Adam and SGD on training and test performance. """
    if adam_best_accuracy > sgd_best_accuracy:
        best_optimizer = "Adam"
        best_number_of_layers = adam_best_hidden_layers
        best_learning_rate = adam_best_learning_rate
        best_neurons = adam_best_neuron_option
    else:
        best_optimizer = "AGD"
        best_number_of_layers = sgd_best_hidden_layers
        best_learning_rate = sgd_best_learning_rate
        best_neurons = sgd_best_neuron_option
    """ Draw Confusion matrix & ROC/AUC curve for different classes """
    print(f'\n\nConfusion matrix & ROC/AUC curve for different classes')
    print(f'--------------------------------------------------------------------------')
    plot_confusion_matrix_and_roc(df=df_abalone, best_optimizer=best_optimizer,
                                  best_number_of_layers=best_number_of_layers,
                                  best_learning_rate=best_learning_rate,
                                  best_neurons=best_neurons)



if __name__ == "__main__":
    main()
