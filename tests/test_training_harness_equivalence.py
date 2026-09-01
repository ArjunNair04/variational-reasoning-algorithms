from reference_kernel_audit import run_audit


def test_frozen_training_kernels_match_reference_package() -> None:
    run_audit()
