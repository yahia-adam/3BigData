.PHONY: all clean classification regression train_all lm clf_lm reg_lm mlp clf_mlp reg_mlp rbf clf_rbf reg_rbf svm clf_svm train_with_datasets train_lm train_mlp train_rbf train_svm

# Linear Model (lm)
CLF_LM_EXAMPLES := \
    lm_classification_cross \
    lm_classification_linear_simple \
    lm_classification_linear_multiple \
    lm_classification_xor

REG_LM_EXAMPLES := \
    lm_regression_linear_simple_2D \
    lm_regression_linear_simple_3D \
    lm_regression_linear_tricky_3D \
    lm_regression_non_linear_simple_2D \
    lm_regression_non_linear_simple_3D

# Multilayer Perceptron (mlp)
CLF_MLP_EXAMPLES := \
    mlp_classification_cross \
    mlp_classification_multi_cross \
    mlp_classification_linear_multiple \
    mlp_classification_linear_simple \
    mlp_classification_multi_linear_3_classes \
    mlp_classification_xor

REG_MLP_EXAMPLES := \
    mlp_regression_linear_simple_2D \
    mlp_regression_linear_simple_3D \
    mlp_regression_linear_tricky_3D \
    mlp_regression_non_linear_simple_2D \
    mlp_regression_non_linear_simple_3D

# Radial Basis Function (rbf)
CLF_RBF_EXAMPLES := \
    rbf_classification_cross \
    rbf_classification_linear_simple \
    rbf_classification_linear_multiple \
    rbf_classification_multi_cross \
    rbf_classification_xor \
    rbf_classification_multi_linear_3_classes

REG_RBF_EXAMPLES := \
    rbf_regression_linear_simple_2D \
    rbf_regression_linear_simple_3D \
    rbf_regression_linear_tricky_3D \
    rbf_regression_non_linear_simple_2D \
    rbf_regression_non_linear_simple_3D

# Support Vector Machine (svm)
CLF_SVM_EXAMPLES := \
    svm_classification_cross \
    svm_classification_linear_simple \
    svm_classification_linear_multiple \
    svm_classification_multi_cross \
    svm_classification_multi_linear_3_classes \
    svm_classification_xor

# Training with Datasets
TRAINING_LM := \
    train_linear_model

TRAINING_MLP := \
    train_multilayer_perceptron

TRAINING_RBF := \
    train_rbf

TRAINING_SVM := \
    train_svm

all: classification regression train_all

classification: clf_lm clf_mlp clf_rbf clf_svm

regression: reg_lm reg_mlp reg_rbf

train_all: train_with_datasets

train_with_datasets: train_lm train_mlp train_rbf train_svm

train_lm:
	@for example in $(TRAINING_LM); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

train_mlp:
	@for example in $(TRAINING_MLP); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

train_rbf:
	@for example in $(TRAINING_RBF); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

train_svm:
	@for example in $(TRAINING_SVM); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

clf_lm:
	@for example in $(CLF_LM_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

reg_lm:
	@for example in $(REG_LM_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

clf_mlp:
	@for example in $(CLF_MLP_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

reg_mlp:
	@for example in $(REG_MLP_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

clf_rbf:
	@for example in $(CLF_RBF_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

reg_rbf:
	@for example in $(REG_RBF_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

clf_svm:
	@for example in $(CLF_SVM_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

clean:
	@cargo clean
