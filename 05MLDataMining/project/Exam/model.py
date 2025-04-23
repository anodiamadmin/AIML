# model.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# Load preprocessed data
data = np.load("viz/processed_data.npz")
X, y = data["X"], data["y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

def build_model(optimizer='adam', l2_val=0.0):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, input_shape=(X.shape[1],), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_val)),
        tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_val)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def evaluate_model(optimizer_name):
    model = build_model(optimizer=optimizer_name)
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"{optimizer_name.upper()} → Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")
    return model, test_acc

# Compare Adam and SGD
adam_model, adam_acc = evaluate_model('adam')
sgd_model, sgd_acc = evaluate_model('sgd')

# Choose best optimizer
best_model = adam_model if adam_acc > sgd_acc else sgd_model
best_opt = 'adam' if adam_acc > sgd_acc else 'sgd'

# L2 regularization
print("\nApplying L2 regularization on best model...")
l2_model = build_model(optimizer=best_opt, l2_val=0.01)
l2_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
l2_test_acc = l2_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"L2 Regularized ({best_opt.upper()}) → Test Acc: {l2_test_acc:.2f}")

# Confusion matrix and ROC/AUC
y_pred = l2_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nConfusion Matrix:\n", confusion_matrix(y_true_classes, y_pred_classes))
print("\nClassification Report:\n", classification_report(y_true_classes, y_pred_classes))

# ROC and AUC
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
print(f"ROC AUC: {roc_auc:.2f}")
RocCurveDisplay.from_predictions(y_test.ravel(), y_pred.ravel())
plt.title("ROC Curve")
plt.show()
