.PHONY: all clean classification regression

CLASSIFICATION_EXAMPLES := \
    mlp_classification_cross \
    mlp_classification_multi_cross \
    mlp_classification_linear_multiple \
    mlp_classification_linear_simple \
    mlp_classification_multi_linear_3_classes \
    mlp_classification_xor

REGRESSION_EXAMPLES := \
    mlp_resgression_linear_simple_2D \
    mlp_regression_linear_simple_3D \
    mlp_regression_linear_tricky_3D \
    mlp_regression_non_linear_simple_2D \
    mlp_regression_non_linear_simple_3D

all: classification regression

classification:
	@for example in $(CLASSIFICATION_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

regression:
	@for example in $(REGRESSION_EXAMPLES); do \
		echo "Running $$example"; \
		cargo run --release --example $$example; \
	done

clean:
	@cargo clean
