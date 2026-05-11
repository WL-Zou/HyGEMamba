#!/usr/bin/env python3
"""Verify that HyGE-Mamba can be imported and run a small forward pass."""

from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check public package imports."""
    print("Checking imports...")
    errors = []

    modules = [
        ("configs.model_config", "ModelConfig"),
        ("configs.training_config", "TrainingConfig"),
        ("src.models.hyge_mamba", "HyGEMamba, HyGEMambaConfig"),
        ("src.modules.common", "RMSNorm, GLUFFN, SSMBlock"),
        ("src.modules.genome", "GenomeBranch, SNPEmbedding"),
        ("src.modules.weather", "WeatherBranch, IntraDayEncoder"),
        ("src.modules.fusion", "GxEFusion"),
        ("src.modules.moe", "SparseMoE, TopKRouter"),
        ("src.modules.heads", "APSIMHead, CurveDecoder, ScalarHead"),
        ("src.data", "HyGEDataset, SyntheticDataGenerator"),
        ("src.losses", "MultiTaskLoss, CurveLoss, MoEAuxiliaryLoss"),
        ("src.utils", "Registry, init_weights, CheckpointManager, get_logger"),
    ]

    for module_path, items in modules:
        try:
            exec(f"from {module_path} import {items}")
            print(f"  [OK] {module_path}")
        except ImportError as exc:
            print(f"  [FAIL] {module_path}: {exc}")
            errors.append((module_path, str(exc)))

    return errors


def check_model_creation():
    """Check that a small model can be constructed."""
    print("\nChecking model creation...")

    try:
        from src.models.hyge_mamba import HyGEMamba, HyGEMambaConfig

        config = HyGEMambaConfig.small()
        print(f"  [OK] Config created: d_model={config.mamba_d_model}")

        model = HyGEMamba(config=config)
        print("  [OK] Model created")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  [OK] Parameters: {n_params:,}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Model creation failed: {exc}")
        return False


def check_synthetic_data():
    """Check synthetic data generation."""
    print("\nChecking synthetic data generation...")

    try:
        from src.data.synthetic_generator import SyntheticDataGenerator, SyntheticDataConfig

        config = SyntheticDataConfig(
            num_samples=5,
            num_snps=50,
            num_days=20,
            num_parents_a=3,
            num_parents_b=3,
        )
        print(f"  [OK] Config created: {config.num_samples} samples")

        generator = SyntheticDataGenerator(config)
        data = generator.generate()
        print("  [OK] Data generated")

        assert data["genotypes_a"].shape == (5, 50), "Wrong genotype shape"
        assert data["hourly_weather"].shape == (5, 20, 24, 8), "Wrong weather shape"
        print("  [OK] Shapes verified")
        return True
    except Exception as exc:
        print(f"  [FAIL] Synthetic data generation failed: {exc}")
        return False


def check_forward_pass():
    """Check a minimal forward pass."""
    print("\nChecking forward pass...")

    try:
        import torch
        from src.models.hyge_mamba import HyGEMamba, HyGEMambaConfig

        config = HyGEMambaConfig(
            snp_embedding_dim=16,
            mamba_d_model=16,
            mamba_d_state=4,
            mamba_num_layers=1,
            num_segments=5,
            intra_day_dim=16,
            inter_day_dim=16,
            fusion_hidden_dim=32,
            fusion_output_dim=32,
            moe_hidden_dim=64,
            num_experts=2,
            top_k=1,
            num_shared_experts=1,
            use_moe=True,
            use_apsim_head=True,
            use_curve_decoder=True,
            num_timesteps=30,
            num_curves=2,
            dropout=0.0,
        )
        model = HyGEMamba(config=config)
        model.eval()
        print("  [OK] Model created")

        batch_size = 2
        num_snps = 1024
        num_days = 30

        genotypes_a = torch.randint(0, 3, (batch_size, num_snps))
        genotypes_b = torch.randint(0, 3, (batch_size, num_snps))
        chromosomes = torch.zeros(batch_size, num_snps, dtype=torch.long)
        positions = torch.arange(num_snps).unsqueeze(0).expand(batch_size, -1)
        hourly_weather = torch.randn(batch_size, num_days, 24, 8)
        print("  [OK] Inputs created")

        with torch.no_grad():
            outputs = model(
                genotypes_a=genotypes_a,
                genotypes_b=genotypes_b,
                chromosomes=chromosomes,
                positions=positions,
                weather=hourly_weather,
            )

        print("  [OK] Forward pass complete")
        assert outputs.yield_pred is not None, "Missing yield output"
        assert outputs.yield_pred.shape == (batch_size,), f"Wrong yield shape: {outputs.yield_pred.shape}"
        print(f"  [OK] Yield output: shape={outputs.yield_pred.shape}")

        if outputs.apsim_params is not None:
            apsim_shapes = {key: value.shape for key, value in outputs.apsim_params.items()}
            print(f"  [OK] APSIM params: shapes={apsim_shapes}")
        if outputs.growth_curves is not None:
            print(f"  [OK] Growth curves: shape={outputs.growth_curves.shape}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Forward pass failed: {exc}")
        traceback.print_exc()
        return False


def check_loss_computation():
    """Check that the multitask loss runs on dummy tensors."""
    print("\nChecking loss computation...")

    try:
        import torch
        from src.losses.multi_task_loss import MultiTaskLoss

        loss_fn = MultiTaskLoss()
        print("  [OK] Loss function created")

        predictions = {
            "yield": torch.randn(4),
            "apsim_params": torch.randn(4, 6),
            "growth_curves": torch.randn(4, 4, 90),
            "scalar_traits": torch.randn(4, 5),
        }
        targets = {
            "yields": torch.randn(4),
            "apsim_params": torch.randn(4, 6),
            "growth_curves": torch.randn(4, 4, 90),
            "scalar_traits": torch.randn(4, 5),
        }

        loss, loss_dict = loss_fn(predictions, targets)
        print(f"  [OK] Loss computed: {loss.item():.4f}")
        print(f"  [OK] Loss components: {list(loss_dict.keys())}")
        return True
    except Exception as exc:
        print(f"  [FAIL] Loss computation failed: {exc}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("HyGE-Mamba Installation Verification")
    print("=" * 60)

    all_passed = True

    if check_imports():
        all_passed = False

    try:
        import torch

        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("\n[WARN] PyTorch not installed. Skipping model tests.")
        print("\nTo install PyTorch:")
        print("  pip install torch")
        return

    checks = [
        check_model_creation,
        check_synthetic_data,
        check_forward_pass,
        check_loss_computation,
    ]
    for check in checks:
        if not check():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] All verification checks passed!")
    else:
        print("[FAIL] Some verification checks failed.")
    print("=" * 60)
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
