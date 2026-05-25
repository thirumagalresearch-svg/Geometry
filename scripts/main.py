"""Entry point for the selfie rectification pipeline."""
import argparse
import sys
from pathlib import Path

from scripts.pipeline import SelfieRectifier


def main() -> None:
    """Run the MVP pipeline on a single image."""
    parser = argparse.ArgumentParser(
        description="Geometry-Aware Selfie Perspective Rectification (RTX 3050 Friendly)"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input selfie image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save final output (defaults to outputs/<timestamp>/final.png)",
    )
    args = parser.parse_args()

    try:
        rectifier = SelfieRectifier()
        print(f"[INFO] Processing: {args.input}")
        output_path = rectifier.process(args.input)
        print(f"\n[OK] Processing complete!")
        print(f"[OK] Final output: {output_path}")
    except FileNotFoundError as e:
        print(f"[ERROR] Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[FATAL] Fatal error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
