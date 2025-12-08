"""
Script to train Graph Tsetlin Machine models for Hex winner prediction.

End-to-end pipeline:
    - Compile Hex C generators if needed
    - Generate raw games (skip if data already exists)
    - Build GTM graph datasets (skip if datasets already exist)
    - Train stage models

Usage examples:
    python scripts/2_train_model.py --board-size 10 --stage all
    python scripts/2_train_model.py --board-size 7 --stage end --num-train 5000 --num-test 1500
"""

import argparse
import os
import sys
import pickle
import shutil
import subprocess
from pathlib import Path

# IMPORTANT: Set CUDA device BEFORE any imports that use CUDA
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(f"INFO: Setting CUDA_VISIBLE_DEVICES=0")
else:
    print(f"INFO: Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import HexGraphTM, Predictor
from src.utils import Config
REPO_ROOT = Path(__file__).resolve().parent.parent
HEX_BIN_DIR = REPO_ROOT / "hex_binaries"
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_STAGES = [0, -2, -5]  # "all states" set


def parse_stages_arg(stages_str: str):
    """Parse stages argument into a list of stage identifiers (strings)."""
    if stages_str.lower() == "all":
        return [str(s) for s in DEFAULT_STAGES]
    return [s.strip() for s in stages_str.split(",") if s.strip()]


def hex_datagen_exe(board_size: int):
    """Return expected hex_datagen executable path for board size."""
    suffix = ".exe" if os.name == "nt" else ""
    return HEX_BIN_DIR / f"hex_datagen_{board_size}x{board_size}{suffix}"


def compile_hex_datagen(board_size: int):
    """Compile hex_datagen_stages for an arbitrary board size."""
    if os.name == "nt":
        # Windows: use cl (requires Developer Command Prompt / Build Tools)
        cl = shutil.which("cl")
        if not cl:
            raise EnvironmentError("cl compiler not found. Please run from VS Developer Prompt.")
        cmd = [
            "cl",
            "/O2",
            f"/DBOARD_DIM={board_size}",
            "hex_datagen_stages.c",
            f"/Fe:hex_datagen_{board_size}x{board_size}.exe",
        ]
        subprocess.run(cmd, check=True, cwd=HEX_BIN_DIR)
    else:
        gcc = shutil.which("gcc")
        if not gcc:
            raise EnvironmentError("gcc not found. Install build-essential.")
        cmd = [
            "gcc",
            "-O3",
            f"-DBOARD_DIM={board_size}",
            "-o",
            f"hex_datagen_{board_size}x{board_size}",
            "hex_datagen_stages.c",
            "-lm",
        ]
        subprocess.run(cmd, check=True, cwd=HEX_BIN_DIR)


def ensure_hex_datagen_compiled(board_size: int):
    """Ensure the hex_datagen executable for the board size exists; compile if missing."""
    exe = hex_datagen_exe(board_size)
    if exe.exists():
        print(f"[OK] Found data generator: {exe}")
        return exe

    print(f"[INFO] Missing data generator {exe}. Compiling on-the-fly for board {board_size}...")
    compile_hex_datagen(board_size)

    if not exe.exists():
        raise FileNotFoundError(f"Expected generator not created: {exe}")

    print(f"[OK] Compiled and found generator: {exe}")
    return exe


def game_npz_paths(board_size: int, data_dir: Path):
    """Return train/test npz paths for generated games."""
    train_npz = data_dir / f"train_games_{board_size}x{board_size}.npz"
    test_npz = data_dir / f"test_games_{board_size}x{board_size}.npz"
    return train_npz, test_npz


def gtm_dataset_paths(board_size: int, stages: list, data_dir: Path):
    """Return expected train/test pkl paths for each stage."""
    paths = []
    for stage in stages:
        paths.append((
            data_dir / f"train_gtm_{board_size}x{board_size}_{stage}.pkl",
            data_dir / f"test_gtm_{board_size}x{board_size}_{stage}.pkl",
        ))
    return paths


def ensure_games_exist(board_size: int, num_train: int, num_test: int, stages_str: str, data_dir: Path):
    """Generate raw games if npz files are missing."""
    train_npz, test_npz = game_npz_paths(board_size, data_dir)
    default_data_dir = REPO_ROOT / "data"
    default_train, default_test = game_npz_paths(board_size, default_data_dir)

    # Reuse default data if present
    if not (train_npz.exists() and test_npz.exists()):
        if data_dir != default_data_dir and default_train.exists() and default_test.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(default_train, train_npz)
            shutil.copy2(default_test, test_npz)
            print(f"[INFO] Copied existing raw games from {default_data_dir} to {data_dir}")

    if train_npz.exists() and test_npz.exists():
        print(f"[SKIP] Raw games already exist at {train_npz} and {test_npz}")
        return

    ensure_hex_datagen_compiled(board_size)

    gen_script = SCRIPTS_DIR / "1_generate_games.py"
    cmd = [
        sys.executable,
        str(gen_script),
        "--board-size",
        str(board_size),
        "--num-train",
        str(num_train),
        "--num-test",
        str(num_test),
        "--save-states",
        stages_str,
    ]

    print(f"[RUN] Generating games: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    # The generator writes to REPO_ROOT/data; copy to custom data_dir if needed
    if data_dir != default_data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
        if default_train.exists() and default_test.exists():
            shutil.copy2(default_train, train_npz)
            shutil.copy2(default_test, test_npz)

    if not (train_npz.exists() and test_npz.exists()):
        raise FileNotFoundError("Game generation did not produce expected npz files.")
    print(f"[OK] Generated raw games at {train_npz} and {test_npz}")


def ensure_gtm_datasets_exist(board_size: int, stages: list, stages_arg: str, data_dir: Path):
    """Build GTM datasets if any stage dataset is missing."""
    expected = gtm_dataset_paths(board_size, stages, data_dir)
    default_data_dir = REPO_ROOT / "data"
    default_expected = gtm_dataset_paths(board_size, stages, default_data_dir)

    # Copy from default location if needed
    if not all(train.exists() and test.exists() for train, test in expected):
        if data_dir != default_data_dir and all(tr.exists() and te.exists() for tr, te in default_expected):
            data_dir.mkdir(parents=True, exist_ok=True)
            for (src_tr, src_te), (dst_tr, dst_te) in zip(default_expected, expected):
                shutil.copy2(src_tr, dst_tr)
                shutil.copy2(src_te, dst_te)
            print(f"[INFO] Copied existing GTM datasets from {default_data_dir} to {data_dir}")

    if all(train.exists() and test.exists() for train, test in expected):
        print("[SKIP] GTM datasets already exist for requested stages.")
        return

    build_script = SCRIPTS_DIR / "1b_build_gtm_datasets.py"
    cmd = [
        sys.executable,
        str(build_script),
        "--board-size",
        str(board_size),
        "--stages",
        stages_arg if stages_arg.lower() != "all" else "all",
    ]
    print(f"[RUN] Building GTM datasets: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # Re-check existence
    # If builder saved to default path, copy back to custom data_dir
    if data_dir != default_data_dir:
        data_dir.mkdir(parents=True, exist_ok=True)
        for (src_tr, src_te), (dst_tr, dst_te) in zip(default_expected, expected):
            if src_tr.exists() and not dst_tr.exists():
                shutil.copy2(src_tr, dst_tr)
            if src_te.exists() and not dst_te.exists():
                shutil.copy2(src_te, dst_te)

    expected = gtm_dataset_paths(board_size, stages, data_dir)
    missing = [(tr, te) for tr, te in expected if not (tr.exists() and te.exists())]
    if missing:
        raise FileNotFoundError(f"GTM dataset build missing files: {missing}")
    print("[OK] GTM datasets ready.")


def get_auto_params(board_size: int, depth: int):
    """Get recommended parameters based on board size and depth."""
    # Clause recommendations
    clause_map = {
        5: 1000,
        6: 1500,
        7: 2000,
        8: 2500,
        9: 3000,
        10: 4000,
        11: 5000,
    }
    clauses = clause_map.get(board_size, 2000 + (board_size - 7) * 500)
    if depth >= 3:
        clauses = int(clauses * 1.5)
    
    # T recommendations
    T_map = {
        5: 15,
        6: 20,
        7: 25,
        8: 30,
        9: 35,
        10: 40,
        11: 50,
    }
    T = T_map.get(board_size, 25)
    
    # s recommendations (tuple for depth > 1)
    if depth == 1:
        s = 3.0
    elif depth == 2:
        s = (4.0, 2.5)
    elif depth == 3:
        s = (5.0, 3.0, 2.0)
    else:
        s = tuple(5.0 - i * 0.7 for i in range(depth))
    
    return clauses, T, s


def load_gtm_dataset(filepath: str):
    """Load a GTM dataset from pickle file."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data['labels'])} samples")
    return data['graphs'], data['labels']


def train_stage(
    stage_name: str,
    config: Config,
    save_model: bool = False
):
    """
    Train a model for a specific game stage.

    Args:
        stage_name: Name of the stage (e.g., 'end', '-2', '-5')
        config: Configuration object
        save_model: Whether to save the trained model
    """
    print("\n" + "="*60)
    print(f"TRAINING MODEL FOR STAGE: {stage_name}")
    print("="*60)

    # Load datasets with fallback for "end" -> "0" naming
    train_path = f"{config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"
    test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_{stage_name}.pkl"

    # Fallback: if stage is "end" and file not found, try "0"
    if not os.path.exists(train_path) and stage_name == "end":
        alt_train_path = f"{config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_0.pkl"
        if os.path.exists(alt_train_path):
            print(f"[INFO] Using alternative naming: '0' instead of 'end'")
            train_path = alt_train_path
            test_path = f"{config.data_dir}/test_gtm_{config.board_size}x{config.board_size}_0.pkl"

    if not os.path.exists(train_path):
        print(f"\nERROR: Training dataset not found at {train_path}")
        if stage_name == "end":
            print(f"Also tried: {config.data_dir}/train_gtm_{config.board_size}x{config.board_size}_0.pkl")
        print("Please run 1b_build_gtm_datasets.py first!")
        return None

    if not os.path.exists(test_path):
        print(f"\nERROR: Test dataset not found at {test_path}")
        print("Please run 1b_build_gtm_datasets.py first!")
        return None

    train_graphs, train_labels = load_gtm_dataset(train_path)
    test_graphs, test_labels = load_gtm_dataset(test_path)

    # Create model
    print("\nInitializing Graph Tsetlin Machine...")
    model = HexGraphTM(
        board_size=config.board_size,  # Added board_size parameter
        number_of_clauses=config.number_of_clauses,
        T=config.T,
        s=config.s,
        depth=config.depth,
        message_size=config.message_size,
        message_bits=config.message_bits,
        max_included_literals=config.max_included_literals,
        grid=config.grid,
        block=config.block
    )

    # Create predictor
    predictor = Predictor(model)

    # Train with validation
    train_acc, test_acc = predictor.train(
        train_graphs=train_graphs,
        train_labels=train_labels,
        test_graphs=test_graphs,
        test_labels=test_labels,
        epochs=config.epochs,
        test_every=config.test_every
    )

    # Detailed evaluation
    train_results = predictor.evaluate_detailed(
        train_graphs,
        train_labels,
        name=f"Training Set (Stage: {stage_name})"
    )

    test_results = predictor.evaluate_detailed(
        test_graphs,
        test_labels,
        name=f"Test Set (Stage: {stage_name})"
    )

    # Save model if requested
    if save_model:
        print("\n[INFO] Saving model...")
        os.makedirs(config.models_dir, exist_ok=True)
        model_path = config.get_model_path(stage_name)
        try:
            saved = model.save(model_path)
            if saved:
                history_path = model_path.replace('.pkl', '_history.pkl')
                predictor.save_training_history(history_path)
                print(f"[OK] Model saved to {model_path}")
            else:
                print(f"[WARN] Model was not saved (PyCUDA state not pickleable).")
        except Exception as e:
            print(f"[WARNING] Could not save model: {e}")
            print("[INFO] Continuing; accuracy numbers are still valid.")

    return {
        'model': model,
        'predictor': predictor,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_results': train_results,
        'test_results': test_results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train GTM models for Hex winner prediction (FIXED VERSION)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT PARAMETER CHANGES:
  The default parameters have been FIXED to use correct values:
  - T: Now 15-50 (was 5000-15000) 
  - s: Now 3.0-5.0 or tuples (was 0.01-10.0)
  - Clauses: Now auto-set 1000-4000 based on board size
  - max_included_literals: Now None (was 255)
  
Examples:
  # Auto-configure for 7x7 board (RECOMMENDED)
  python scripts/2_train_model.py --board-size 7 --stage end --auto-params
  
  # Manual configuration for experiments
  python scripts/2_train_model.py --board-size 7 --stage end --clauses 2000 --T 25 --s 4.0 --depth 2
        """
    )

    parser.add_argument('--board-size', type=int, default=10,
                        help='Size of the Hex board (default: 10)')
    parser.add_argument('--stage', type=str, default='all',
                        help='Stage to train: end, -2, -5, or all (default: all)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing GTM datasets (default: data)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save models (default: models)')
    parser.add_argument('--num-train', type=int, default=100000,
                        help='Number of training games to generate if missing (default: 100000)')
    parser.add_argument('--num-test', type=int, default=30000,
                        help='Number of test games to generate if missing (default: 30000)')
    parser.add_argument('--gen-stages', type=str, default='all',
                        help='Stages to generate/build datasets for (default: all -> 0,-2,-5)')

    # Auto-configuration
    parser.add_argument('--auto-params', action='store_true',
                        help='Use automatic parameter configuration based on board size (RECOMMENDED)')

    # Training parameters with FIXED defaults
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--clauses', type=int, default=None,
                        help='Number of clauses (default: auto-set based on board size)')
    parser.add_argument('--depth', type=int, default=2,
                        help='Message passing depth (default: 2, was 6)')
    parser.add_argument('--T', type=int, default=None,
                        help='Threshold T (default: auto-set 15-50, was 5000)')
    parser.add_argument('--s', type=float, default=None,
                        help='Specificity s - single value (default: auto-set based on depth)')
    
    # Advanced s configuration
    parser.add_argument('--s-tuple', type=str, default=None,
                        help='Specificity as tuple for multi-depth, e.g., "4.0,2.5" for depth=2')

    # Advanced parameters
    parser.add_argument('--message-size', type=int, default=256,
                        help='Message size (default: 256)')
    parser.add_argument('--message-bits', type=int, default=2,
                        help='Message bits (default: 2)')
    parser.add_argument('--max-included-literals', type=int, default=None,
                        help='Max included literals (default: None = no limit, was 255)')
    parser.add_argument('--test-every', type=int, default=5,
                        help='Test every N epochs (default: 5)')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.board_size = args.board_size
    config.data_dir = args.data_dir
    config.models_dir = args.models_dir
    config.epochs = args.epochs
    config.test_every = args.test_every
    config.message_size = args.message_size
    config.message_bits = args.message_bits
    data_dir = (REPO_ROOT / config.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine depth
    depth = args.depth

    # Auto-params or manual configuration
    if args.auto_params or (args.clauses is None and args.T is None and args.s is None):
        print("\n[AUTO-CONFIGURATION] Using recommended parameters for board size", args.board_size)
        auto_clauses, auto_T, auto_s = get_auto_params(args.board_size, depth)
        
        config.number_of_clauses = auto_clauses
        config.T = auto_T
        config.s = auto_s
        config.depth = depth
        config.max_included_literals = None
        
        print(f"  Auto-configured:")
        print(f"    Clauses: {auto_clauses}")
        print(f"    T: {auto_T}")
        print(f"    s: {auto_s}")
        print(f"    depth: {depth}")
        print(f"    max_included_literals: None")
    else:
        # Manual configuration with validation
        config.number_of_clauses = args.clauses if args.clauses is not None else get_auto_params(args.board_size, depth)[0]
        config.T = args.T if args.T is not None else get_auto_params(args.board_size, depth)[1]
        config.depth = depth
        config.max_included_literals = args.max_included_literals
        
        # Handle s parameter (can be single float or tuple)
        if args.s_tuple:
            # Parse tuple from string "4.0,2.5"
            config.s = tuple(float(x.strip()) for x in args.s_tuple.split(','))
            print(f"[INFO] Using s as tuple: {config.s}")
        elif args.s is not None:
            config.s = args.s
        else:
            config.s = get_auto_params(args.board_size, depth)[2]
        
        # Validate parameters
        if config.T > 100:
            print(f"\n[WARNING] T={config.T} is very high! Recommended: 15-50")
            print(f"          This may prevent the model from learning properly.")
            print(f"          Consider using --auto-params or setting T=15-50")
        
        if isinstance(config.s, float) and config.s > 20:
            print(f"\n[WARNING] s={config.s} is very high! Recommended: 3.0-5.0")
            print(f"          This may cause unstable learning.")
        
        if config.number_of_clauses < 500:
            print(f"\n[WARNING] {config.number_of_clauses} clauses may be too few!")
            print(f"          Recommended: 1000-4000 for board size {args.board_size}")

    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    config.print_config()
    print("="*60)

    # Prepare stages for data generation/building
    generation_stages = parse_stages_arg(args.gen_stages)
    generation_stage_arg = args.gen_stages  # original string for subprocess
    print(f"\nRequested generation stages: {generation_stages}")

    # Ensure data exists (skip if already present)
    ensure_games_exist(
        board_size=args.board_size,
        num_train=args.num_train,
        num_test=args.num_test,
        stages_str=generation_stage_arg if generation_stage_arg.lower() != "all" else ",".join(generation_stages),
        data_dir=data_dir,
    )

    ensure_gtm_datasets_exist(
        board_size=args.board_size,
        stages=generation_stages,
        stages_arg=generation_stage_arg,
        data_dir=data_dir,
    )

    # Determine which stages to train
    if args.stage == 'all':
        stages = ['end', '-2', '-5']
    else:
        stages = [args.stage]

    print(f"\nTraining stages: {stages}\n")

    # Train each stage
    results = {}
    for stage in stages:
        result = train_stage(stage, config, save_model=True)
        if result is not None:
            results[stage] = result

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for stage, result in results.items():
        print(f"\nStage: {stage}")
        print(f"  Training Accuracy: {result['train_acc']:.2f}%")
        print(f"  Test Accuracy: {result['test_acc']:.2f}%")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nYou can now evaluate the models:")
    print(f"  python scripts/3_evaluate.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()