

# import libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score
import os
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping callback
from tensorflow.keras.callbacks import ReduceLROnPlateau # Keep import in case it's used elsewhere, but not needed for custom loop LR reduction
from google.colab import drive # Import drive here

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score

import matplotlib.pyplot as plt # Import matplotlib for plotting


from tensorflow.keras.models import save_model

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Define your Google Drive base path (adjust as needed)
GOOGLE_DRIVE_PATH = '/content/drive/MyDrive/datasets/'

# Training parameters
BATCH_SIZE = 5
NUM_EPOCHS = 10

# Parameters for AMRL (initial values – these will be trained)
M_POS_INIT = 1.0
M_NEG_INIT = -1.0
LAMBDA_POS_INIT = 2.0
LAMBDA_NEG_INIT = 1.0
BETA_POS_INIT = 2
BETA_NEG_INIT = 1

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 20  # Number of epochs with no improvement before stopping
MIN_DELTA = 0.0001            # Minimum improvement to be considered as progress

# Gradient clipping parameter
CLIP_NORM = 1.0  # Maximum L2 norm of the gradients

# EVALUATION METRICS

"""#FALSE_ALARM_RATE(FAR)=Ratio of the number of false prediction of danger and the total number of the model prediction when actually no danger was observed.
                                  #FAR=b/(b+d);0=<FAR<=1"""
def false_alarm(tp,tn,fp,fn):

    false_alarm = fp / (fp + tn)
    false_alarm = round(false_alarm, 2)

#print('The false alarm value of model is:',false_alarm)
    return false_alarm




"""#BIAS_SCORE=Ratio of the number of days whe the model predicted 'danger and the total number of avalanche occurences events.
                                  #B=(a+b)/(a+c)
                                  #B=1 -> Unbiased
                                  #B<=1 -> Underforcast
                                  #B>=1 -> Overforecast"""
def bias_score(tp,tn,fp,fn):

    bias_score = (tp + fp) / (tp + fn)
    bias_score = round(bias_score, 2)
    #print('The bias score of model is:',bias_score)
    return bias_score




"""#Probability of detection(PoD)=Probability that the event was forecast when it occurred.
                                  #PoD=a/(a+c);0=<Pod<=1"""
def probability_of_detection(tp,tn,fp,fn):
    pod_value = tp / (tp + fn)
    pod_value = round(pod_value, 2)
    #print('The POD value of model is:',pod_value)
    return pod_value




"""#Heidke_skill_score(HSS)=Skill score based on hit rate:HSS=(Hit + Correct Negative - Chance)/(Total - Chance), where chance is the expected number of correct event forecast due to chance.
                            #HSS=2(ad-bc)/((a+b)(b+d)+(a+c)(c+d)); (-)infinity=<HSS<=1 """
def heidke_skill_score(tp,tn,fp,fn):
    heidke_skill_value = 2 * ((tp * tn) - (fn * fp)) /(((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn)))
    heidke_skill_value = round(heidke_skill_value,2)
    #print('The HSS score of model is:',heidke_skill_value)
    return heidke_skill_value


"""#Pierce_skill_score(PSS)=Skill score based on hit rate:HSS=(Hit + Correct Negative - Chance)/(Total - Chance), where chance is the expected number of correct event forecast due to chance.
                            #HSS=2(ad-bc)/((a+b)(b+d)+(a+c)(c+d)); (-)infinity=<HSS<=1 """
def pierce_skill_score(tp,tn,fp,fn):
    pierce_skill_value = (tp / (tp + fn)) - (fp / (tp + tn))
    pierce_skill_value = round(pierce_skill_value,2)
    #print('The PSS score of model is:',pierce_skill_value)
    return pierce_skill_value

def f1_score(tp,tn,fp,fn):
    f1_value = (2*(tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp)) + (tp/(tp+fn)))
    f1_value = round(f1_value, 2)
    #print('The F1-Score of the model is:',f1_value)
    return f1_value

# Function to compute class ratios
def get_class_ratio(y, decimals=2):
    counts = np.bincount(y)
    total = len(y)
    # Ensure counts has at least two elements for binary classification
    if len(counts) < 2:
        if 0 in y:
            counts = np.array([counts[0], 0])
        else:
            counts = np.array([0, counts[0]])

    ratio = {cls: round(counts[cls] / total, decimals) for cls in range(len(counts))}

    # Compute imbalance ratio (majority/minority)
    majority_count = max(counts)
    minority_count = min(counts)
    if minority_count > 0:
        imbalance_ratio = f"{round(majority_count / minority_count, decimals)} : 1"
    elif majority_count > 0:
         imbalance_ratio = "inf : 1" # All samples are majority
    else:
         imbalance_ratio = "0 : 0" # No samples


    return counts, ratio, imbalance_ratio

# Function to plot training history
def plot_training_history(history, dataset_name, loss_type, save_dir):
    """
    Plots training and validation metrics from the history object.
    Saves the plots to a specified directory.
    """
    # Handle cases where history might be empty or None
    if not history:
        print(f"Warning: No history data available to plot for {loss_type} on {dataset_name}.")
        return

    # Get epochs from one of the history lists (assuming all have the same length)
    # Check if any list in history is non-empty before getting length
    first_key = next(iter(history), None)
    if not first_key or not history[first_key]:
         print(f"Warning: History dictionary is empty or its lists are empty for {loss_type} on {dataset_name}. Cannot plot.")
         return

    epochs = range(1, len(history[first_key]) + 1)

    # Define metrics to plot based on availability in history keys
    metrics_to_plot = [key for key in history if not key.startswith('val_') and key != 'epoch']
    val_metrics_to_plot = [key for key in history if key.startswith('val_')]


    for metric_name in metrics_to_plot:
        val_metric_name = f'val_{metric_name}'
        if val_metric_name in history:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, history[metric_name], 'b', label=f'Training {metric_name.replace("_", " ")}')
            plt.plot(epochs, history[val_metric_name], 'r', label=f'Validation {val_metric_name.replace("val_", "").replace("_", " ")}')
            plt.title(f'{metric_name.replace("_", " ").capitalize()} for {loss_type} on {dataset_name}')
            plt.xlabel('Epochs')
            plt.ylabel(metric_name.replace("_", " ").capitalize())
            plt.legend()
            plt.grid(True)

            # Sanitize filename
            safe_dataset_name = dataset_name.replace('.xlsx', '').replace(' ', '_').replace('-', '_')
            safe_loss_type = loss_type.replace('-', '_')
            filename = f"{safe_dataset_name}_{safe_loss_type}_{metric_name}_plot.png"
            save_path = os.path.join(save_dir, filename)

            try:
                plt.savefig(save_path)
                print(f"Saved plot to: {save_path}")
            except Exception as e:
                print(f"Error saving plot {filename}: {e}")

            plt.close() # Close the figure to free up memory

# Custom train step for AMRL with gradient clipping and lambda clipping
# Removed @tf.function decorator
def train_step_amrl(model, loss_function, optimizer, X_batch, y_batch, clip_norm):
    with tf.GradientTape() as tape:
        y_pred_logits = model(X_batch, training=True) # Set training=True for dropout
        loss = loss_function(y_batch, y_pred_logits)

    # Get gradients for both model's trainable variables AND loss function's trainable variables
    # Ensure all trainable variables are collected
    trainable_variables = model.trainable_variables + loss_function.trainable_variables if hasattr(loss_function, 'trainable_variables') else model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)

    # Check for None gradients and filter
    if any(grad is None for grad in gradients):
        none_grads = [var.name for grad, var in zip(gradients, trainable_variables) if grad is None]
        # print(f"Warning: Gradients are None for variables: {none_grads}") # Keep this for debugging if needed
        grads_and_vars = [(grad, var) for grad, var in zip(gradients, trainable_variables) if grad is not None]
        if not grads_and_vars:
             # This should ideally not happen if loss and model are connected
            raise ValueError("All gradients are None. Cannot apply gradients.")
    else:
        grads_and_vars = list(zip(gradients, trainable_variables))


    # Apply gradient clipping
    clipped_gradients, _ = tf.clip_by_global_norm([g for g, v in grads_and_vars], clip_norm)

    # Apply gradients
    optimizer.apply_gradients(zip(clipped_gradients, [v for g, v in grads_and_vars]))

    # Manually clip lambda values to be non-negative for AMRLLossTF after applying gradients
    if isinstance(loss_function, AMRLLossTF):
         if hasattr(loss_function, 'lambda1') and hasattr(loss_function, 'lambda0'):
            loss_function.lambda1.assign(tf.maximum(0., loss_function.lambda1.numpy()))
            loss_function.lambda0.assign(tf.maximum(0., loss_function.lambda0.numpy()))


    return loss

# Custom test step for AMRL (used for validation)
# Removed @tf.function decorator
def test_step_amrl(model, loss_function, X_batch, y_batch, metrics):
    y_pred_logits = model(X_batch, training=False) # Set training=False for dropout
    loss = loss_function(y_batch, y_pred_logits)

    # Update metrics
    y_pred_probs = tf.sigmoid(y_pred_logits)
    metrics['accuracy'].update_state(y_batch, y_pred_probs)
    metrics['precision'].update_state(y_batch, y_pred_probs)
    metrics['recall'].update_state(y_batch, y_pred_probs)
    metrics['auc_roc'].update_state(y_batch, y_pred_probs)
    metrics['auc_pr'].update_state(y_batch, y_pred_probs)

    return loss


# Custom training loop for AMRL
def custom_train_loop(model, loss_function, optimizer, train_dataset, val_dataset, epochs, early_stopping, clip_norm):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    # Define paths for saving temporary weights
    model_weights_path = 'temp_model_weights.weights.h5' # Corrected filename extension
    loss_weights_path = 'temp_loss_weights.npy'


    history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'auc_roc': [], 'auc_pr': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_auc_roc': [], 'val_auc_pr': []
    }

    # Initialize metrics for tracking within the custom loop
    train_metrics = {
        'accuracy': BinaryAccuracy(threshold=0.5),
        'precision': Precision(thresholds=0.5),
        'recall': Recall(thresholds=0.5),
        'auc_roc': AUC(),
        'auc_pr': AUC(curve='PR')
    }
    val_metrics = {
        'accuracy': BinaryAccuracy(threshold=0.5),
        'precision': Precision(thresholds=0.5),
        'recall': Recall(thresholds=0.5),
        'auc_roc': AUC(),
        'auc_pr': AUC(curve='PR')
    }


    for epoch in range(epochs):
        # print(f"\nEpoch {epoch+1}/{epochs}") # Reduced verbosity during batch processing

        # Reset metrics at the start of each epoch
        for metric in train_metrics.values():
            metric.reset_state()
        for metric in val_metrics.values():
            metric.reset_state()

        total_train_loss = 0
        num_train_batches = 0
        for X_batch, y_batch in train_dataset:
            loss = train_step_amrl(model, loss_function, optimizer, X_batch, y_batch, clip_norm)
            total_train_loss += loss.numpy()
            num_train_batches += 1

            # Update train metrics (need to manually update here as train_step doesn't return probabilities/predictions)
            # Use training=False to get deterministic predictions for metrics evaluation on the training data batch
            y_pred_logits = model(X_batch, training=False)
            y_pred_probs = tf.sigmoid(y_pred_logits)
            train_metrics['accuracy'].update_state(y_batch, y_pred_probs)
            train_metrics['precision'].update_state(y_batch, y_pred_probs)
            train_metrics['recall'].update_state(y_batch, y_pred_probs)
            train_metrics['auc_roc'].update_state(y_batch, y_pred_probs)
            train_metrics['auc_pr'].update_state(y_batch, y_pred_probs)


        avg_train_loss = total_train_loss / num_train_batches

        total_val_loss = 0
        num_val_batches = 0
        for X_batch_val, y_batch_val in val_dataset:
            loss = test_step_amrl(model, loss_function, X_batch_val, y_batch_val, val_metrics)
            total_val_loss += loss.numpy()
            num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        # Get metric results
        train_acc = train_metrics['accuracy'].result().numpy()
        train_prec = train_metrics['precision'].result().numpy()
        train_recall = train_metrics['recall'].result().numpy()
        train_auc_roc = train_metrics['auc_roc'].result().numpy()
        train_auc_pr = train_metrics['auc_pr'].result().numpy()

        val_acc = val_metrics['accuracy'].result().numpy()
        val_prec = val_metrics['precision'].result().numpy()
        val_recall = val_metrics['recall'].result().numpy()
        val_auc_roc = val_metrics['auc_roc'].result().numpy()
        val_auc_pr = val_metrics['auc_pr'].result().numpy()

        # Append metrics to history for plotting
        history['loss'].append(avg_train_loss)
        history['accuracy'].append(train_acc)
        history['precision'].append(train_prec)
        history['recall'].append(train_recall)
        history['auc_roc'].append(train_auc_roc)
        history['auc_pr'].append(train_auc_pr)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_recall)
        history['val_auc_roc'].append(val_auc_roc)
        history['val_auc_pr'].append(val_auc_pr)


        # Print epoch summary (reduced verbosity)
        # print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Prec: {train_prec:.4f}, Train Recall: {train_recall:.4f}, Train AUC-ROC: {train_auc_roc:.4f}, Train AUC-PR: {train_auc_pr:.4f}")
        # print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val Recall: {val_recall:.4f}, Val AUC-ROC: {val_auc_roc:.4f}, Val AUC-PR: {val_auc_pr:.4f}")


        # Print the current values of the trainable loss parameters (only for AMRL)
        if isinstance(loss_function, AMRLLossTF):
            learned_params = [loss_function.m1.numpy(), loss_function.m0.numpy(), loss_function.lambda1.numpy(), loss_function.lambda0.numpy()]
            # print(f"  AMRL Learned Params: m1={learned_params[0]:.4f}, m0={learned_params[1]:.4f}, lambda1={learned_params[2]:.4f}, lambda0={learned_params[3]:.4f}") # Reduced verbosity


        # --- Early Stopping Check (using validation loss) ---
        # Check if validation loss improved by at least min_delta
        if avg_val_loss < best_val_loss - early_stopping.min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save current weights if validation loss improved
            model.save_weights(model_weights_path)
            # Save loss function trainable variables (only for AMRL)
            if isinstance(loss_function, AMRLLossTF):
                 np.save(loss_weights_path, learned_params)
            # print("Saved model and loss weights.") # Reduced verbosity

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping.patience:
                print(f"Early stopping triggered based on validation loss after {epoch+1} epochs.")
                # Restore best weights if early stopping is triggered
                if os.path.exists(model_weights_path):
                    print("Restoring model weights from best validation epoch...")
                    model.load_weights(model_weights_path)
                # Restore loss parameters (only for AMRL)
                if isinstance(loss_function, AMRLLossTF) and os.path.exists(loss_weights_path):
                     print("Restoring loss parameters from best validation epoch...")
                     best_loss_params = np.load(loss_weights_path)
                     loss_function.m1.assign(best_loss_params[0])
                     loss_function.m0.assign(best_loss_params[1])
                     loss_function.lambda1.assign(best_loss_params[2])
                     loss_function.lambda0.assign(best_loss_params[3])
                     # print(f"  Restored AMRL Params: m1={loss_function.m1.numpy():.4f}, m0={loss_function.m0.numpy():.4f}, lambda1={loss_function.lambda1.numpy():.4f}, lambda0={loss_function.lambda0.numpy():.4f}") # Reduced verbosity

                break # Exit the training loop

    return history # Return the history dictionary

class AMRLLossTF(tf.keras.losses.Loss):
    """
    Margin Loss for Imbalanced Datasets in TensorFlow.
    Combines Weighted Binary Cross-Entropy with an Asymmetric Margin Penalty.
    """
    def __init__(self, w1, w0, m1_init, m0_init, lambda1_init, lambda0_init, beta1=1, beta0=1, name="margin_loss_tf"):
        super().__init__(name=name)
        self.w1 = tf.Variable(w1, dtype=tf.float32, trainable=False) # BCE weights are not trainable
        self.w0 = tf.Variable(w0, dtype=tf.float32, trainable=False) # BCE weights are not trainable
        # Make margins and lambdas trainable variables as per original design intent
        self.m1 = tf.Variable(m1_init, dtype=tf.float32, trainable=True, name="m1")
        self.m0 = tf.Variable(m0_init, dtype=tf.float32, trainable=True, name="m0")
        self.lambda1 = tf.Variable(lambda1_init, dtype=tf.float32, trainable=True, name="lambda1")
        self.lambda0 = tf.Variable(lambda0_init, dtype=tf.float32, trainable=True, name="lambda0")
        self.beta1 = tf.constant(float(beta1), dtype=tf.float32) # Betas are constants
        self.beta0 = tf.constant(float(beta0), dtype=tf.float32) # Betas are constants

    def call(self, y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_logits = tf.cast(y_pred_logits, tf.float32)

        bce_unweighted = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)
        l_w_bce = tf.where(tf.equal(y_true, 1), bce_unweighted * self.w1, bce_unweighted * self.w0)

        l_amp = tf.zeros_like(y_pred_logits)
        penalty_minority = self.lambda1 * tf.pow(tf.maximum(0., self.m1 - y_pred_logits), self.beta1)
        l_amp = tf.where(tf.equal(y_true, 1), penalty_minority, l_amp)

        penalty_majority = self.lambda0 * tf.pow(tf.maximum(0., y_pred_logits - self.m0), self.beta0)
        l_amp = tf.where(tf.equal(y_true, 0), penalty_majority, l_amp)

        total_loss = tf.reduce_mean(l_w_bce + l_amp)

        # Compare individual components:

        #print("Weighted BCE:", tf.reduce_mean(l_w_bce).numpy(), "AMP:", tf.reduce_mean(l_amp).numpy())

        #print("Margins and Lambdas:", self.m1.numpy(), self.m0.numpy(), self.lambda1.numpy(), self.lambda0.numpy())



        return total_loss

    @property
    def trainable_variables(self):
        # Explicitly list the trainable variables of the loss function
        return [self.m1, self.m0, self.lambda1, self.lambda0]




# Custom train step for PGFL with gradient clipping
# Removed @tf.function decorator
def train_step_pgfl(model, loss_function, optimizer, X_batch, y_batch, clip_norm):
    with tf.GradientTape() as tape:
        y_pred_logits = model(X_batch, training=True) # Set training=True for dropout
        loss = loss_function(y_batch, y_pred_logits) # loss.call calculates loss

    # Update moving averages *after* calculating the loss and outside the gradient tape
    # This ensures the update uses the gradients from the loss calculation
    y_pred_probs = tf.sigmoid(y_pred_logits)
    if isinstance(loss_function, PGFLossTF): # Only update for PGFLossTF
        loss_function.update_confidence_moving_averages(y_batch, y_pred_probs)


    # Get gradients for model's trainable variables AND loss function's trainable variable (alpha)
    trainable_variables = model.trainable_variables + loss_function.trainable_variables if hasattr(loss_function, 'trainable_variables') else model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)

    # Add a check for None gradients and filter
    if any(grad is None for grad in gradients):
        none_grads = [var.name for grad, var in zip(gradients, trainable_variables) if grad is None]
        # print(f"Warning: Gradients are None for variables: {none_grads}") # Keep this for debugging if needed
        grads_and_vars = [(grad, var) for grad, var in zip(gradients, trainable_variables) if grad is not None]
        if not grads_and_vars:
            raise ValueError("All gradients are None. Cannot apply gradients.")
    else:
        grads_and_vars = list(zip(gradients, trainable_variables))


    # Apply gradient clipping
    clipped_gradients, _ = tf.clip_by_global_norm([g for g, v in grads_and_vars], clip_norm)

    # Apply gradients
    optimizer.apply_gradients(zip(clipped_gradients, [v for g, v in grads_and_vars]))


    return loss

# Custom test step for PGFL (used for validation)
# Removed @tf.function decorator
def test_step_pgfl(model, loss_function, X_batch, y_batch, metrics):
    y_pred_logits = model(X_batch, training=False) # Set training=False for dropout
    loss = loss_function(y_batch, y_pred_logits) # loss.call updates MC0/MC1 (can stay in call for test, or move out)

    # Update metrics
    y_pred_probs = tf.sigmoid(y_pred_logits)
    metrics['accuracy'].update_state(y_batch, y_pred_probs)
    metrics['precision'].update_state(y_batch, y_pred_probs)
    metrics['recall'].update_state(y_batch, y_pred_probs)
    metrics['auc_roc'].update_state(y_batch, y_pred_probs)
    metrics['auc_pr'].update_state(y_batch, y_pred_probs)

    return loss

# Custom training loop for PGFL
def custom_train_loop_pgfl(model, loss_function, optimizer, train_dataset, val_dataset, epochs, early_stopping, clip_norm):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    # Define paths for saving temporary weights for both models
    model_weights_path = 'temp_model_weights_pgfl.weights.h5' # Separate paths for PGFL
    loss_weights_path = 'temp_loss_weights_pgfl.npy' # Separate paths for PGFL

    history = {
        'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'auc_roc': [], 'auc_pr': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_auc_roc': [], 'val_auc_pr': []
    }

    train_metrics = {
        'accuracy': BinaryAccuracy(threshold=0.5), 'precision': Precision(thresholds=0.5),
        'recall': Recall(thresholds=0.5), 'auc_roc': AUC(), 'auc_pr': AUC(curve='PR')
    }
    val_metrics = {
        'accuracy': BinaryAccuracy(threshold=0.5), 'precision': Precision(thresholds=0.5),
        'recall': Recall(thresholds=0.5), 'auc_roc': AUC(), 'auc_pr': AUC(curve='PR')
    }


    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        for metric in train_metrics.values(): metric.reset_state()
        for metric in val_metrics.values(): metric.reset_state()
        # Reset loss function's internal metrics (MC0, MC1) at the start of each epoch?
        # Or let them accumulate? The description suggests moving averages over time,
        # so perhaps don't reset per epoch, but per file. The re-initialization per file handles this.
        # loss_function.reset_metrics() # Only if resetting per epoch is desired


        total_train_loss = 0
        num_train_batches = 0
        for i, (X_batch, y_batch) in enumerate(train_dataset): # Add enumerate to get batch index
            loss = train_step_pgfl(model, loss_function, optimizer, X_batch, y_batch, clip_norm)
            total_train_loss += loss.numpy()
            num_train_batches += 1
            # print(f"  Batch {i}: Loss = {loss.numpy():.4f}") # Print loss per batch


            # Update train metrics manually
            y_pred_logits = model(X_batch, training=False)
            y_pred_probs = tf.sigmoid(y_pred_logits)
            train_metrics['accuracy'].update_state(y_batch, y_pred_probs)
            train_metrics['precision'].update_state(y_batch, y_pred_probs)
            train_metrics['recall'].update_state(y_batch, y_pred_probs)
            train_metrics['auc_roc'].update_state(y_batch, y_pred_probs)
            train_metrics['auc_pr'].update_state(y_batch, y_pred_probs)


        avg_train_loss = total_train_loss / num_train_batches

        total_val_loss = 0
        num_val_batches = 0
        for X_batch_val, y_batch_val in val_dataset:
            loss = test_step_pgfl(model, loss_function, X_batch_val, y_batch_val, val_metrics)
            total_val_loss += loss.numpy()
            num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        train_acc = train_metrics['accuracy'].result().numpy()
        train_prec = train_metrics['precision'].result().numpy()
        train_recall = train_metrics['recall'].result().numpy()
        train_auc_roc = train_metrics['auc_roc'].result().numpy()
        train_auc_pr = train_metrics['auc_pr'].result().numpy()

        val_acc = val_metrics['accuracy'].result().numpy()
        val_prec = val_metrics['precision'].result().numpy()
        val_recall = val_metrics['recall'].result().numpy()
        val_auc_roc = val_metrics['auc_roc'].result().numpy()
        val_auc_pr = val_metrics['auc_pr'].result().numpy()

        history['loss'].append(avg_train_loss)
        history['accuracy'].append(train_acc)
        history['precision'].append(train_prec)
        history['recall'].append(train_recall)
        history['auc_roc'].append(train_auc_roc)
        history['auc_pr'].append(train_auc_pr)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_recall)
        history['val_auc_roc'].append(val_auc_roc)
        history['val_auc_pr'].append(val_auc_pr)


        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Prec: {train_prec:.4f}, Train Recall: {train_recall:.4f}, Train AUC-ROC: {train_auc_roc:.4f}, Train AUC-PR: {train_auc_pr:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Prec: {val_prec:.4f}, Val Recall: {val_recall:.4f}, Val AUC-ROC: {val_auc_roc:.4f}, Val AUC-PR: {val_auc_pr:.4f}")

        # Print the current value of the trainable alpha and moving averages
        if isinstance(loss_function, PGFLossTF):
             print(f"  PGFL Learned Alpha: {loss_function.alpha.numpy():.4f}, MC0: {loss_function.mc0.numpy():.4f}, MC1: {loss_function.mc1.numpy():.4f}")


        # --- Early Stopping Check (using validation loss) ---
        if avg_val_loss < best_val_loss - early_stopping.min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save current weights if validation loss improved
            model.save_weights(model_weights_path)
            # Save loss function trainable variables (alpha) and moving averages
            np.save(loss_weights_path, [loss_function.alpha.numpy(), loss_function.mc0.numpy(), loss_function.mc1.numpy()])
            #print("Saved model and PGFL loss parameters.")

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping.patience:
                print(f"Early stopping triggered based on validation loss after {epoch+1} epochs.")
                # Restore best weights if early stopping is triggered
                if os.path.exists(model_weights_path):
                    print("Restoring model weights from best validation epoch...")
                    model.load_weights(model_weights_path)
                if os.path.exists(loss_weights_path):
                    print("Restoring PGFL loss parameters from best validation epoch...")
                    best_loss_params = np.load(loss_weights_path)
                    loss_function.alpha.assign(best_loss_params[0])
                    loss_function.mc0.assign(best_loss_params[1])
                    loss_function.mc1.assign(best_loss_params[2])
                    print(f"  Restored PGFL Params: Alpha={loss_function.alpha.numpy():.4f}, MC0={loss_function.mc0.numpy():.4f}, MC1={loss_function.mc1.numpy():.4f}")

                break # Exit the training loop

    return history # Return the history dictionary

class PGFLossTF(tf.keras.losses.Loss):
    """
    Proposed Geometric Focal Loss (PGFL) for Imbalanced Datasets in TensorFlow.
    Combines dynamic class weighting based on confidence with Focal Loss.
    """
    def __init__(self, num_majority, num_minority, gamma=2.0, alpha_init=None, name="pgf_loss_tf"):
        super().__init__(name=name)
        self.gamma = tf.constant(gamma, dtype=tf.float32)

        # Initialize alpha (adaptive weight for minority class)
        # If alpha_init is not provided, use initial inverse frequency
        if alpha_init is None:
            initial_alpha = tf.cast(num_majority / (num_majority + num_minority), dtype=tf.float32)
        else:
            initial_alpha = tf.constant(alpha_init, dtype=tf.float32)

        # Make alpha a trainable variable for dynamic adaptation
        self.alpha = tf.Variable(initial_alpha, dtype=tf.float32, trainable=True, name="alpha")

        # Moving averages for confidence (non-trainable)
        self.mc0 = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="mc0") # Confidence for correctly classified majority
        self.mc1 = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="mc1") # Confidence for correctly classified minority
        self.momentum = tf.constant(0.99, dtype=tf.float32) # Momentum for moving averages

        # Track counts for updating moving averages (non-trainable)
        self.count0 = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="count0")
        self.count1 = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="count1")


    def update_confidence_moving_averages(self, y_true, y_pred_probs):
        """Updates the moving averages of confidence for correctly classified samples."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred_probs = tf.cast(y_pred_probs, tf.float32)

        # Identify correctly classified samples
        predicted_labels = tf.cast(y_pred_probs > 0.5, tf.float32)
        correctly_classified_mask = tf.equal(predicted_labels, y_true)

        # Separate based on true class
        majority_mask = tf.equal(y_true, 0)
        minority_mask = tf.equal(y_true, 1)

        # Correctly classified majority samples
        cc_majority_mask = tf.logical_and(correctly_classified_mask, majority_mask)
        cc_majority_probs = tf.boolean_mask(1.0 - y_pred_probs, cc_majority_mask) # Confidence for majority is 1 - prob of minority
        num_cc_majority = tf.cast(tf.shape(cc_majority_probs)[0], tf.float32)

        # Correctly classified minority samples
        cc_minority_mask = tf.logical_and(correctly_classified_mask, minority_mask)
        cc_minority_probs = tf.boolean_mask(y_pred_probs, cc_minority_mask) # Confidence for minority is prob of minority
        num_cc_minority = tf.cast(tf.shape(cc_minority_probs)[0], tf.float32)


        # Update moving averages if there are correctly classified samples
        if num_cc_majority > 0:
            avg_cc_majority_confidence = tf.reduce_mean(cc_majority_probs)
            self.mc0.assign(self.momentum * self.mc0 + (1.0 - self.momentum) * avg_cc_majority_confidence)
            self.count0.assign(self.count0 + num_cc_majority)

        if num_cc_minority > 0:
            avg_cc_minority_confidence = tf.reduce_mean(cc_minority_probs)
            self.mc1.assign(self.momentum * self.mc1 + (1.0 - self.momentum) * avg_cc_minority_confidence)
            self.count1.assign(self.count1 + num_cc_minority)


    def update_alpha_dynamically(self):
        """Dynamically updates alpha based on the moving averages of confidence."""
        # Avoid division by zero if no samples of a class have been correctly classified yet
        epsilon = tf.keras.backend.epsilon()


        confidence_ratio = (self.mc1 + epsilon) / (self.mc0 + epsilon)


        scaled_confidence_difference = 1.0 * (self.mc0 - self.mc1) # Adjust scaling factor as needed


        pass # No direct update to self.alpha here

    def call(self, y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_logits = tf.cast(y_pred_logits, tf.float32)

        # Calculate probabilities
        p = tf.sigmoid(y_pred_logits)


        self.update_confidence_moving_averages(y_true, p)


        # Calculate standard BCE
        bce_unweighted = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)

        # p_t is the probability of the true class
        p_t = y_true * p + (1 - y_true) * (1 - p)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # Adaptive alpha_t: Use the trainable self.alpha
        # alpha_t is alpha for positive (minority), 1-alpha for negative (majority)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1.0 - self.alpha)

        # PGFL Loss: alpha_t * (1 - p_t)^gamma * BCE
        pgf_loss = alpha_t * modulating_factor * bce_unweighted

        # Add print statements for debugging
        # tf.print("y_true:", y_true, summarize=-1)
        # tf.print("y_pred_logits:", y_pred_logits, summarize=-1)
        # tf.print("p:", p, summarize=-1)
        # tf.print("bce_unweighted:", bce_unweighted, summarize=-1)
        # tf.print("p_t:", p_t, summarize=-1)
        # tf.print("modulating_factor:", modulating_factor, summarize=-1)
        # tf.print("alpha_t:", alpha_t, summarize=-1)
        # tf.print("pgf_loss (per sample):", pgf_loss, summarize=-1)
        # tf.print("pgf_loss (mean):", tf.reduce_mean(pgf_loss), summarize=-1)

        return tf.reduce_mean(pgf_loss)

    @property
    def trainable_variables(self):
        # The only trainable variable in this loss is alpha
        return [self.alpha]

    def reset_metrics(self):
        """Resets the state of the moving average variables."""
        self.mc0.assign(0.0)
        self.mc1.assign(0.0)
        self.count0.assign(0.0)
        self.count1.assign(0.0)


# --- 2. Data Loading from Google Drive ---
def load_data_from_drive(file_path):
    """
    Loads a dataset from a given file path.
    Assumes the last column is the target and no header row in the CSV file.
    Args:
        file_path (str): The full path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}.")
        return None

    # Read CSV assuming with header. P
    # Assuming the files are Excel (.xlsx) as per the main execution block
    try:
        df = pd.read_excel(file_path)


        # Drop columns by name if they exist
        df = df.drop(columns=['Date', 'Index'], errors='ignore')

        print(f"DataFrame after dropping 'Date' and 'Index' columns: {df.shape}")
        print(f"Remaining columns: {df.columns.tolist()}")

        # Show class distribution of the assumed target (last column)
        print(f"Assuming the last column '{df.columns[-1]}' is the target column.")
        print(f"Class distribution:\n{df.iloc[:, -1].value_counts()}")

        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

class ClassBalancedLossTF(tf.keras.losses.Loss):
    """
    Class Balanced Loss for Imbalanced Datasets in TensorFlow.
    Re-weights the standard loss by the 'effective number of samples' for each class.
    """
    def __init__(self, num_majority, num_minority, beta=0.999, name="class_balanced_loss_tf"):
        super().__init__(name=name)
        self.beta = tf.constant(beta, dtype=tf.float32)

        # Calculate effective number of samples for each class
        # Add a small epsilon to avoid division by zero in case num_samples is 0 or beta is 1
        epsilon = tf.keras.backend.epsilon()

        effective_num_majority = (1.0 - tf.pow(self.beta, tf.cast(num_majority, tf.float32))) / (1.0 - self.beta + epsilon)
        effective_num_minority = (1.0 - tf.pow(self.beta, tf.cast(num_minority, tf.float32))) / (1.0 - self.beta + epsilon)

        # Calculate class-balanced weights
        self.w0_cb = tf.constant(1.0, dtype=tf.float32) / effective_num_majority
        self.w1_cb = tf.constant(1.0, dtype=tf.float32) / effective_num_minority

        tf.print(f"CB Loss Weights: w0 (majority)={self.w0_cb}, w1 (minority)={self.w1_cb}")

    def call(self, y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_logits = tf.cast(y_pred_logits, tf.float32)

        # Calculate standard BCE
        bce_unweighted = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)

        # Apply class-balanced weights
        cb_loss = tf.where(tf.equal(y_true, 1), bce_unweighted * self.w1_cb, bce_unweighted * self.w0_cb)

        return tf.reduce_mean(cb_loss)

class FocalLossTF(tf.keras.losses.Loss):
    """
    Focal Loss for Imbalanced Datasets in TensorFlow.
    Focuses training on hard, misclassified examples by down-weighting easy examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, name="focal_loss_tf"):
        super().__init__(name=name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.gamma = tf.constant(gamma, dtype=tf.float32)

    def call(self, y_true, y_pred_logits):
        y_true = tf.cast(y_true, tf.float32)
        y_pred_logits = tf.cast(y_pred_logits, tf.float32)

        # Calculate standard BCE (log loss)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred_logits)

        # Calculate probabilities
        p = tf.sigmoid(y_pred_logits)

        # p_t is the probability of the true class
        p_t = y_true * p + (1 - y_true) * (1 - p)

        # alpha_t is alpha for positive, 1-alpha for negative
        alpha_t = y_true * self.alpha + (1 - y_true) * (1.0 - self.alpha)

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        # Focal Loss: alpha_t * (1 - p_t)^gamma * BCE
        focal_loss = alpha_t * modulating_factor * bce
        return tf.reduce_mean(focal_loss)

# Define the plots save directory again as it's needed for the print statement
plots_save_dir = os.path.join(GOOGLE_DRIVE_PATH, 'training_plots')


# Define a function for model initialization
def initialize_model(input_dim, loss_type, num_majority_train, num_minority_train):
    model = SimpleMLP_TF(input_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Optimizer for the main model (Generator)


    loss_function = None # Classification loss function 

    if loss_type == 'AMRL':
        w1_bce_amrl = num_majority_train / (num_minority_train + num_majority_train)
        w0_bce_amrl = num_minority_train / (num_minority_train + num_majority_train)
        loss_function = AMRLLossTF(w1 = w1_bce_amrl, w0 = w0_bce_amrl, m1_init = M_POS_INIT, m0_init = M_NEG_INIT,
                                  lambda1_init=LAMBDA_POS_INIT, lambda0_init = LAMBDA_NEG_INIT,
                                  beta1 = BETA_POS_INIT, beta0 = BETA_NEG_INIT)
    elif loss_type == 'PGFL':
        alpha_init_pgfl = num_majority_train / (num_majority_train + num_minority_train)
        loss_function = PGFLossTF(num_majority=num_majority_train, num_minority=num_minority_train,
                                  gamma=gamma_pgfl, alpha_init=alpha_init_pgfl)

    elif loss_type in ['BCE', 'FOCAL', 'CB']:
        metrics_list = [
            BinaryAccuracy(name = 'accuracy', threshold = 0.5),
            Precision(name = 'precision', thresholds = 0.5),
            Recall(name = 'recall', thresholds = 0.5),
            AUC(name = 'auc_roc'),
            AUC(curve = 'PR', name ='auc_pr')
        ]
        if loss_type == 'BCE':
            loss_function = tf.keras.losses.BinaryCrossentropy(from_logits = True)
        elif loss_type == 'FOCAL':
            alpha_focal = num_majority_train / (num_minority_train + num_minority_train)
            gamma_focal = 2.0
            loss_function = FocalLossTF(alpha = alpha_focal, gamma = gamma_focal)
        elif loss_type == 'CB':
            beta_cb = 0.999
            loss_function = ClassBalancedLossTF(num_majority = num_majority_train, num_minority = num_minority_train, beta = beta_cb)

        # Compile the model for Keras's fit method
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics_list)


    return model, optimizer, loss_function, discriminator_model, discriminator_optimizer


# Define a function for training the model
def train_model(model, optimizer, loss_function, discriminator_model, discriminator_optimizer,
                train_dataset, val_dataset, epochs, early_stopping_callback, clip_norm, loss_type):
    history = None
    if loss_type == 'AMRL':
        print("\nStarting custom training loop for AMRL...")
        history = custom_train_loop(model, loss_function, optimizer, train_dataset, val_dataset, epochs, early_stopping_callback, clip_norm)
        print("Custom training loop finished.")
    elif loss_type == 'PGFL':
        print("\nStarting custom training loop for PGFL...")
        history = custom_train_loop_pgfl(model, loss_function, optimizer, train_dataset, val_dataset, epochs, early_stopping_callback, CLIP_NORM)
        print("Custom training loop finished.")

    elif loss_type in ['BCE', 'FOCAL', 'CB']:
        print("\nStarting model.fit training...")
        # The model is already compiled in initialize_model for these loss types
        history = model.fit(train_dataset.unbatch().map(lambda x, y: (x, y)).batch(BATCH_SIZE), # Re-batch for model.fit
                            epochs=epochs,
                            validation_data=val_dataset.unbatch().map(lambda x, y: (x, y)).batch(BATCH_SIZE), # Re-batch for model.fit
                            callbacks=[early_stopping_callback],
                            verbose=1)
        history = history.history
        print("model.fit training finished.")

    return history


# Define a function for evaluating the model and collecting results
def evaluate_model(model, loss_type, history, X_test_scaled, y_test_tf, y_test, BATCH_SIZE):
    print("\nEvaluating on the Test Set...")

    # Evaluate the model using model.evaluate or manual calculation for custom loop
    # Calculate metrics manually using predictions for all loss types for consistency
    y_pred_logits_test = model.predict(X_test_scaled, batch_size=BATCH_SIZE)
    y_pred_probs_test = tf.sigmoid(y_pred_logits_test).numpy().flatten()
    y_pred_binary_test = (y_pred_probs_test > 0.5).astype(int)
    y_true_test_flat = y_test.values.flatten()

    # 1. Get predicted probabilities from model
    y_pred_probs = model.predict(X_test_scaled, batch_size=BATCH_SIZE)

    # 2. Convert probabilities to class predictions (threshold = 0.5)
    y_pred_labels = (y_pred_probs >= 0.5).astype(int)

    # 3. Ensure y_test is in correct format (binary integers)
    y_true = np.array(y_test_tf).astype(int)

    # 4. Calculate metrics
    accuracy = round(accuracy_score(y_true, y_pred_labels),2)
    precision = round(precision_score(y_true, y_pred_labels, zero_division=0),2)
    recall = round(recall_score(y_true, y_pred_labels, zero_division=0),2)
    auc_roc = round(roc_auc_score(y_true, y_pred_probs),2)
    auc_pr = round(average_precision_score(y_true, y_pred_probs),2)

    # Confusion Matrix components
    tn, fp, fn, tp = confusion_matrix(y_true_test_flat, y_pred_binary_test, labels=[0, 1]).ravel()

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Geometric Mean (G-mean)
    current_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_gmean = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(current_recall * specificity_gmean) if current_recall >= 0 and specificity_gmean >= 0 else np.nan
    g_mean = round(g_mean, 2)


    # Balanced Accuracy
    bal_accuracy = balanced_accuracy_score(y_true_test_flat, y_pred_binary_test)
    bal_accuracy = round(bal_accuracy, 2)

    # Kuipers Skill Score (KSS) / True Skill Score (TSS)
    kss_tss = current_recall + specificity - 1
    kss_tss = round(kss_tss, 2)

    # Heidke Skill Score (HSS)
    far = false_alarm(tp,tn,fp,fn)
    bias = bias_score(tp,tn,fp,fn)
    pod = probability_of_detection(tp,tn,fp,fn)
    hss= heidke_skill_score(tp,tn,fp,fn)
    pss = pierce_skill_score(tp,tn,fp,fn)
    f1  =  f1_score(tp,tn,fp,fn)

    num_minority_test = np.sum(y_test)
    num_majority_test = len(y_test) - num_minority_test # Corrected from num_minority_test

    #Combine Keras metrics and manually calculated metrics
    test_metrics_dict = {
        'Dataset': os.path.basename(current_file_name), # Use the current file name
        'Loss_Type': loss_type,
        #'Loss': loss, # Loss is not directly comparable across loss functions on the test set
        
    
        'Batch_Size': BATCH_SIZE,
        'Positive': num_minority_test,
        'Negative': num_majority_test,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Accuracy' : accuracy,
        'Precision': precision,
        'Recall': recall,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Specificity': specificity,
        'G-Mean': g_mean,
        'Balanced_Accuracy': bal_accuracy,
        'KSS_TSS': kss_tss,
        'HSS': hss,
        'PSS': pss,
        'FAR': far,
        'bias': bias,
        'pod': pod,
        'f1': f1
    }


    # If the loss type is AMRL, add the learned parameters to the results dictionary
    if loss_type == 'AMRL':
         # Need to access the loss function object from the training loop
         # This requires passing it or retrieving it if it's a global/accessible variable
         # For now, assuming loss_function is still available from the training scope
         if 'loss_function' in locals() and isinstance(loss_function, AMRLLossTF):
            test_metrics_dict['Learned_M_POS'] = loss_function.m1.numpy()
            test_metrics_dict['Learned_M_NEG'] = loss_function.m0.numpy()
            test_metrics_dict['Learned_LAMBDA_POS'] = loss_function.lambda1.numpy()
            test_metrics_dict['Learned_LAMBDA_NEG'] = loss_function.lambda0.numpy()

    # If the loss type is PGFL, add the learned alpha and final moving averages
    elif loss_type == 'PGFL':
         if 'loss_function' in locals() and isinstance(loss_function, PGFLossTF):
            test_metrics_dict['Learned_Alpha'] = loss_function.alpha.numpy()
            test_metrics_dict['Final_MC0'] = loss_function.mc0.numpy()
            test_metrics_dict['Final_MC1'] = loss_function.mc1.numpy()
    

    for metric, value in test_metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return test_metrics_dict, history # Return history as well for plotting


if __name__ == "__main__":

    LOSS_TYPES_TO_COMPARE = ['AMRL', 'BCE', 'FOCAL', 'CB', 'PGFL']


    # --- Configuration for data loading ---
    # Ensure GOOGLE_DRIVE_PATH is set and exists
    if 'GOOGLE_DRIVE_PATH' not in globals() or not os.path.exists(GOOGLE_DRIVE_PATH):
        # This check should now pass after the previous cell
        raise ValueError("GOOGLE_DRIVE_PATH not set or does not exist. Please configure it to your Google Drive folder containing XLSX files.")

    print(f"Scanning for XLSX files in: {GOOGLE_DRIVE_PATH}")

    file_paths_to_process = [os.path.join(GOOGLE_DRIVE_PATH, f) for f in os.listdir(GOOGLE_DRIVE_PATH) if f.endswith('.xlsx')]
    # sort the files
    file_paths_to_process = sorted(file_paths_to_process, key=lambda x: os.path.basename(x).lower())

    if not file_paths_to_process:
        print(f"No XLSX files found in {GOOGLE_DRIVE_PATH}. Please check the path and file extensions.")
        # exit() # Don't exit in a single cell, just print message and continue

    # Create directories to save plots and text files within the Google Drive path
    plots_save_dir = os.path.join(GOOGLE_DRIVE_PATH, 'training_plots')
    os.makedirs(plots_save_dir, exist_ok=True)
    print(f"Plots will be saved to: {plots_save_dir}")

    results_save_dir = os.path.join(GOOGLE_DRIVE_PATH, 'text files')
    os.makedirs(results_save_dir, exist_ok=True)
    print(f"Results will be saved to: {results_save_dir}")


    for file_path in file_paths_to_process:
        current_file_name = os.path.basename(file_path)
        print(f"\n############################################################")
        print(f"Processing Dataset: {current_file_name}")
        print(f"############################################################")

        # Load data for the current file
        df = load_data_from_drive(file_path)
        if df is None:
            print(f"Skipping {current_file_name} due to loading error.")
            continue # Skip to the next file

        # Define features (X) and target (y)
        # Assumes the last column is always the target

        # Remove certain columns whihc i think are not relevant
        print(df.head())


        # Drop rows with any NaN values
        df_new = df_new.dropna()

        # Ensure the target is the last column after dropping
        X = df_new.iloc[:, :-1]
        y = df_new.iloc[:, -1]


        # --- Data Preprocessing and Splitting (for current dataset) ---
        print("\nSplitting data into train, validation, and test sets...")
        # Use the processed X and y for splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify = y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = (0.1/0.8), random_state=42, stratify = y_train)

        print(f"Train set size: {len(X_train)} (Minority: {y_train.sum()})")
        print(f"Validation set size: {len(X_val)} (Minority: {y_val.sum()})")
        print(f"Test set size: {len(X_test)} (Minority: {y_test.sum()})") # Corrected to y_test.sum()


        # Collect ratios
        ratios_results = []
        for name, labels in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
          counts, ratio, imbalance_ratio = get_class_ratio(labels.values.flatten()) # Ensure labels are flattened
          ratio_str = ", ".join([f"{cls}: {ratio[cls]:.{2}f}" for cls in ratio])
          ratios_results.append(
            f"{name} set:\n"
            f" Counts: {counts.tolist()} (0s={counts[0]}, 1s={counts[1]})\n" # Access counts correctly
            f" Ratios: {{{ratio_str}}}\n"
            f" Imbalance Ratio (maj:min): {imbalance_ratio}\n"
            )

        # Save to text file
        ratios_file_path = os.path.join(results_save_dir, f"{current_file_name}_class_ratios.txt")
        print(f"Saving class ratios to: {ratios_file_path}")
        with open(ratios_file_path, "w") as f:
          for res in ratios_results:
            f.write(res + "\n")



        scaler = StandardScaler()
        # Apply scaler to the numerical features
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        y_train_tf = y_train.values.astype(np.float32).reshape(-1, 1)
        y_val_tf = y_val.values.astype(np.float32).reshape(-1, 1)
        y_test_tf = y_test.values.astype(np.float32).reshape(-1, 1)


        # Create TensorFlow Datasets for custom training loop
        train_dataset_tf = tf.data.Dataset.from_tensor_slices((X_train_scaled.astype(np.float32), y_train_tf)).batch(BATCH_SIZE).shuffle(buffer_size=len(X_train_scaled))
        
        # test_dataset_tf = tf.data.Dataset.from_tensor_slices((X_test_scaled.astype(np.float32), y_test_tf)).batch(BATCH_SIZE)

        # --- Loss Function Parameters (common calculations for current dataset) ---
        num_minority_train = np.sum(y_train)
        num_majority_train = len(y_train) - num_minority_train


        w1_bce_amrl = num_majority_train / (num_minority_train + num_majority_train)
        w0_bce_amrl = num_minority_train / (num_minority_train + num_majority_train)

        # Parameters for Focal Loss
        alpha_focal = num_majority_train / (num_majority_train + num_minority_train) # Corrected denominator
        gamma_focal = 2.0

        # Parameters for Class Balanced Loss
        beta_cb = 0.999

        # Parameters for PGFL
        gamma_pgfl = 2.0
        # Initialize alpha for PGFL using inverse frequency from training data
        alpha_init_pgfl = num_majority_train / (num_majority_train + num_minority_train)


        # Calculate input_dim from the scaled training data shape
        input_dim = X_train_scaled.shape[1]
        print(f"Input dimension for the model: {input_dim}")

        loss_results_for_file = [] # List to store results for each loss function for the current file

        for loss_type in LOSS_TYPES_TO_COMPARE:
            print(f"\n--------------------------------------------------------")
            print(f"Training {current_file_name} with {loss_type} Loss")
            print(f"--------------------------------------------------------")

            # Re-initialize the model(s) and optimizers for each loss function to ensure a fresh start
            model = SimpleMLP_TF(input_dim)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Optimizer for the main model (Generator)


            loss_function = None # Classification loss function 
            history = None # Initialize history variable


            if loss_type == 'AMRL':
                print("Margin Loss Parameters (Initial Values):")
                print(f"  BCE Weights: w1_bce (minority) = {w1_bce_amrl:.4f}, w0_bce (majority) = {w0_bce_amrl:.4f}")
                print(f"  Margins: m1 (pos) = {M_POS_INIT}, m0 (neg) = {M_NEG_INIT}")
                print(f"  Lambdas: lambda1 (pos) = {LAMBDA_POS_INIT}, lambda0 (neg) = {LAMBDA_NEG_INIT}")
                print(f"  Betas: beta1 (pos) = {BETA_POS_INIT}, beta0 = {BETA_NEG_INIT}")
                loss_function = AMRLLossTF(w1 = w1_bce_amrl, w0 = w0_bce_amrl, m1_init = M_POS_INIT, m0_init = M_NEG_INIT,
                                          lambda1_init=LAMBDA_POS_INIT, lambda0_init = LAMBDA_NEG_INIT,
                                          beta1 = BETA_POS_INIT, beta0 = BETA_NEG_INIT)

                print("\nStarting custom training loop for AMRL...")
                early_stopping_callback = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)
                history = custom_train_loop(model, loss_function, optimizer, train_dataset_tf, val_dataset_inner, NUM_EPOCHS, early_stopping_callback, CLIP_NORM) # Use val_dataset_inner for validation
                print("Custom training loop finished.")

            elif loss_type == 'PGFL':
                print("PGFL Loss Parameters (Initial Values):")
                print(f"  Gamma: {gamma_pgfl:.4f}")
                print(f"  Initial Alpha (Minority Weight): {alpha_init_pgfl:.4f}")
                loss_function = PGFLossTF(num_majority=num_majority_train, num_minority=num_minority_train,
                                          gamma=gamma_pgfl, alpha_init=alpha_init_pgfl)


                print("\nStarting custom training loop for PGFL...")
                early_stopping_callback = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)
                history = custom_train_loop_pgfl(model, loss_function, optimizer, train_dataset_tf, val_dataset_inner, NUM_EPOCHS, early_stopping_callback, CLIP_NORM) # Use val_dataset_inner for validation
                print("Custom training loop finished.")


            else: # Use model.compile and model.fit for other loss functions (BCE, FOCAL, CB)
                metrics_list = [
                    BinaryAccuracy(name = 'accuracy', threshold = 0.5),
                    Precision(name = 'precision', thresholds = 0.5),
                    Recall(name = 'recall', thresholds = 0.5),
                    AUC(name = 'auc_roc'),
                    AUC(curve = 'PR', name ='auc_pr')
                ]

                if loss_type == 'BCE':
                    print("Standard Binary Crossentropy Loss")
                    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits = True)
                elif loss_type == 'FOCAL':
                    print(f"Focal Loss Parameters: alpha={alpha_focal:.4f}, gamma={gamma_focal:.4f}")
                    loss_function = FocalLossTF(alpha = alpha_focal, gamma = gamma_focal)
                elif loss_type == 'CB':
                    print(f"Class Balanced Loss Parameters: beta={beta_cb:.4f}")
                    loss_function = ClassBalancedLossTF(num_majority = num_majority_train, num_minority = num_minority_train, beta = beta_cb)
                else:
                    raise ValueError(f"Invalid LOSS_TYPE: {loss_type}. Choose from 'AMRL', 'BCE', 'FOCAL', 'CB', 'PGFL'")


                # Compile the model using the selected loss function and metrics
                model.compile(optimizer=optimizer, # Use the same optimizer
                              loss=loss_function,
                              metrics=metrics_list)

                # Define Early Stopping callback for model.fit
                early_stopping_callback = EarlyStopping(monitor='val_loss',
                                               patience=EARLY_STOPPING_PATIENCE,
                                               min_delta=MIN_DELTA,
                                               restore_best_weights=True,
                                               verbose=1)

                print("\nStarting model.fit training...")
                history = model.fit(X_train_scaled, y_train_tf,
                                    epochs=NUM_EPOCHS,
                                    batch_size=BATCH_SIZE,
                                    validation_data=(X_val_scaled, y_val_tf), # Use the original val set for model.fit
                                    callbacks=[early_stopping_callback],
                                    verbose=1)
                history = history.history

                print("model.fit training finished.")

            # --- Plotting Training History ---
            if history: # Check if history is not None
                print("\nPlotting training history...")


                # For other losses, plot standard metrics including validation loss
                plot_history = {k: history[k] for k in history if k in ['loss', 'accuracy', 'precision', 'recall', 'auc_roc', 'auc_pr',
                                                                            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_auc_roc', 'val_auc_pr']}
                plot_training_history(plot_history, current_file_name, loss_type, plots_save_dir)


            print("\nEvaluating on the Test Set...")

            print("Train class distribution:", np.bincount(y_train.values.flatten()))
            print("Val class distribution:", np.bincount(y_val.values.flatten()))
            print("Test class distribution:", np.bincount(y_test.values.flatten()))


            # Evaluate the model using model.evaluate or manual calculation for custom loop
            # Calculate metrics manually using predictions for all loss types for consistency
            y_pred_logits_test = model.predict(X_test_scaled, batch_size=BATCH_SIZE)
            y_pred_probs_test = tf.sigmoid(y_pred_logits_test).numpy().flatten()
            y_pred_binary_test = (y_pred_probs_test > 0.5).astype(int)
            y_true_test_flat = y_test.values.flatten()

            # 1. Get predicted probabilities from model
            y_pred_probs = model.predict(X_test_scaled, batch_size=BATCH_SIZE)

            # 2. Convert probabilities to class predictions (threshold = 0.5)
            y_pred_labels = (y_pred_probs >= 0.5).astype(int)

            # 3. Ensure y_test is in correct format (binary integers)
            y_true = np.array(y_test_tf).astype(int)

            # 4. Calculate metrics
            accuracy = round(accuracy_score(y_true, y_pred_labels),2)
            precision = round(precision_score(y_true, y_pred_labels, zero_division=0),2)
            recall = round(recall_score(y_true, y_pred_labels, zero_division=0),2)
            auc_roc = round(roc_auc_score(y_true, y_pred_probs),2)
            auc_pr = round(average_precision_score(y_true, y_pred_probs),2)

            # Confusion Matrix components
            tn, fp, fn, tp = confusion_matrix(y_true_test_flat, y_pred_binary_test, labels=[0, 1]).ravel()

            # Specificity (True Negative Rate)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Geometric Mean (G-mean)
            current_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity_gmean = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            g_mean = np.sqrt(current_recall * specificity_gmean) if current_recall >= 0 and specificity_gmean >= 0 else np.nan
            g_mean = round(g_mean, 2)


            # Balanced Accuracy
            bal_accuracy = balanced_accuracy_score(y_true_test_flat, y_pred_binary_test)
            bal_accuracy = round(bal_accuracy, 2)

            # Kuipers Skill Score (KSS) / True Skill Score (TSS)
            kss_tss = current_recall + specificity - 1
            kss_tss = round(kss_tss, 2)

            # Heidke Skill Score (HSS)
            far = false_alarm(tp,tn,fp,fn)
            bias = bias_score(tp,tn,fp,fn)
            pod = probability_of_detection(tp,tn,fp,fn)
            hss= heidke_skill_score(tp,tn,fp,fn)
            pss = pierce_skill_score(tp,tn,fp,fn)
            f1  =  f1_score(tp,tn,fp,fn)

            num_minority_test = np.sum(y_test)
            num_majority_test = len(y_test) - num_minority_test

            #Combine Keras metrics and manually calculated metrics
            test_metrics_dict = {
                'Dataset': current_file_name,
                'Loss_Type': loss_type,
                
                'Positive': num_minority_test,
                'Negative': num_majority_test,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Accuracy' : accuracy,
                'Precision': precision,
                'Recall': recall,
                'AUC-ROC': auc_roc,
                'AUC-PR': auc_pr,
                'Specificity': specificity,
                'G-Mean': g_mean,
                'Balanced_Accuracy': bal_accuracy,
                'KSS_TSS': kss_tss,
                'HSS': hss,
                'PSS': pss,
                'FAR': far,
                'bias': bias,
                'pod': pod,
                'f1': f1
            }


            # If the loss type is AMRL, add the learned parameters to the results dictionary
            if loss_type == 'AMRL':
                 if isinstance(loss_function, AMRLLossTF):
                    test_metrics_dict['Learned_M_POS'] = loss_function.m1.numpy()
                    test_metrics_dict['Learned_M_NEG'] = loss_function.m0.numpy()
                    test_metrics_dict['Learned_LAMBDA_POS'] = loss_function.lambda1.numpy()
                    test_metrics_dict['Learned_LAMBDA_NEG'] = loss_function.lambda0.numpy()

            # If the loss type is PGFL, add the learned alpha and final moving averages
            elif loss_type == 'PGFL':
                 if isinstance(loss_function, PGFLossTF):
                    test_metrics_dict['Learned_Alpha'] = loss_function.alpha.numpy()
                    test_metrics_dict['Final_MC0'] = loss_function.mc0.numpy()
                    test_metrics_dict['Final_MC1'] = loss_function.mc1.numpy()


            for metric, value in test_metrics_dict.items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")

            # --- Consolidate and Save All Results ---
            # Save this loss result in the list
            loss_results_for_file.append(test_metrics_dict.copy())

        # --- Save all loss results for the current file ---
        file_results_df = pd.DataFrame(loss_results_for_file)
        output_filename = f"{current_file_name}_all_losses.xlsx"
        results_save_dir = os.path.join(GOOGLE_DRIVE_PATH, 'text files')
        os.makedirs(results_save_dir, exist_ok=True)
        print(f"Results will be saved to: {results_save_dir}")
        output_filepath = os.path.join(results_save_dir, output_filename) # Use GOOGLE_DRIVE_PATH for saving
        file_results_df.to_excel(output_filepath, index=False)
        # output_filepath = os.path.join(GOOGLE_DRIVE_PATH, output_filename)
        # file_results_df.to_csv(output_filepath, index=False)

        print(f"\nResults for {current_file_name} saved to: {output_filepath}")
        print("\nComprehensive Comparison Summary for this file:")
        print(file_results_df.round(4).to_string())

# Assuming the results files are saved in the 'text files' directory within GOOGLE_DRIVE_PATH

drive.mount('/content/drive', force_remount=True)

GOOGLE_DRIVE_PATH = '/content/drive/MyDrive/1. Trial/'
results_save_dir = os.path.join(GOOGLE_DRIVE_PATH, 'text files')

# List all the result files in the directory
result_files = [f for f in os.listdir(results_save_dir) if f.endswith('_all_losses.xlsx')]

if not result_files:
    print(f"No result files found in {results_save_dir}. Please ensure the main execution block ran successfully.")
else:
    all_results_list = []
    for file_name in result_files:
        file_path = os.path.join(results_save_dir, file_name)
        try:
            df_results = pd.read_excel(file_path)
            all_results_list.append(df_results)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    if all_results_list:
        # Concatenate all results into a single DataFrame
        all_results_df = pd.concat(all_results_list, ignore_index=True)

        print("\n--- Performance Summary Across Datasets and Loss Functions ---")

        # Group by Loss_Type and calculate mean for key metrics
        # Select key metrics for summary
        summary_metrics = ['Accuracy', 'Precision', 'Recall', 'AUC-ROC', 'AUC-PR',
                           'Specificity', 'G-Mean', 'Balanced_Accuracy', 'KSS_TSS', 'HSS', 'PSS', 'FAR', 'bias', 'pod', 'f1']

        # Ensure only existing columns are included in summary_metrics
        summary_metrics = [metric for metric in summary_metrics if metric in all_results_df.columns]


        performance_summary = all_results_df.groupby('Loss_Type')[summary_metrics].mean().round(4)

        print("Average Performance Metrics per Loss Function (averaged across datasets):")
        display(performance_summary)

        # Optional: Also show metrics per dataset per loss function if needed
        # print("\nPerformance Metrics per Dataset per Loss Function:")
        # display(all_results_df[['Dataset', 'Loss_Type'] + summary_metrics].round(4))

    else:
        print("No data available to summarize after attempting to read result files.")
