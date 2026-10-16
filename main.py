"""CLI entrypoint to run the benchmark evaluation or inference on custom images."""

import argparse
import sys
from predict import run_inference
from src.config import load_config
from src.logger import get_logger, setup_logger
from classification import run_classification_demonstration


def main() -> int:
    """Parses command-line arguments and routes execution to benchmark or inference mode."""
    parser = argparse.ArgumentParser(
        description="Leafy Vegetable Detection & Size Classification System."
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default=None,
        help="Path to unseen image, folder, or ZIP archive. If omitted, runs benchmark evaluation.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml).",
    )
    parser.add_argument(
        "--evaluate",
        "-e",
        action="store_true",
        help="Run Step 6 evaluation audit on unseen test dataset and generate contact sheet.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the interactive Web UI (browser drag-and-drop interface).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Custom output directory for predictions (optional).",
    )

    args = parser.parse_args()

    # Initialize centralized system logger
    try:
        cfg = load_config(args.config)
        setup_logger(
            log_dir=cfg.system.log_dir,
            log_file_name=cfg.system.log_file_name,
            log_level=cfg.system.log_level,
        )
    except Exception:
        setup_logger(log_dir="logs", log_file_name="leafy_cv.log")

    logger = get_logger("main")
    logger.info("Initializing Leafy Vegetable Classification System...")

    if args.ui:
        import app
        logger.info("Launching LeafyCV Web UI at http://127.0.0.1:5000...")
        app.start_server()
        return 0
    elif args.evaluate:
        from evaluate import run_evaluation_pipeline
        logger.info("Routing to Step 6 Unseen Test Evaluation Pipeline...")
        return run_evaluation_pipeline(config_path=args.config)
    elif args.source:
        logger.info(f"Routing to inference runner with source: {args.source}")
        return run_inference(
            source_path=args.source,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    else:
        logger.info("Routing to benchmark classification demonstration...")
        return run_classification_demonstration()


if __name__ == "__main__":
    sys.exit(main())
