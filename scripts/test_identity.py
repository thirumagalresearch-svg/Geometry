import sys
from pathlib import Path
from PIL import Image

# Add root directory to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.pipeline import SelfieRectifier
from scripts.utils import save_image

def main():
    input_path = Path("outputs/gradio/input.png")
    if not input_path.exists():
        print(f"[ERROR] Input image not found at {input_path}")
        sys.exit(1)

    print(f"[INFO] Found input image: {input_path}")
    rectifier = SelfieRectifier()

    brain_dir = Path(r"C:\Users\Dell\.gemini\antigravity\brain\016ce1ef-0c3e-444a-8496-e1cf2df44a2f")
    brain_dir.mkdir(parents=True, exist_ok=True)

    # We will test 4 scenarios:
    # 1. Warp only (No CodeFormer, No Diffusion)
    # 2. CodeFormer with 1.0 fidelity (Max identity preservation, No Diffusion)
    # 3. CodeFormer with 0.9 fidelity (Very high identity preservation, No Diffusion)
    # 4. CodeFormer with 0.8 fidelity (High identity preservation, No Diffusion)

    scenarios = [
        {"name": "warp_only", "cf": False, "cf_w": 1.0, "sd": False, "sd_s": 0.0},
        {"name": "cf_1_0", "cf": True, "cf_w": 1.0, "sd": False, "sd_s": 0.0},
        {"name": "cf_0_9", "cf": True, "cf_w": 0.9, "sd": False, "sd_s": 0.0},
        {"name": "cf_0_8", "cf": True, "cf_w": 0.8, "sd": False, "sd_s": 0.0},
    ]

    for sc in scenarios:
        name = sc["name"]
        print(f"\n--- Running scenario: {name} ---")
        try:
            res = rectifier.process_with_intermediates(
                input_path,
                fidelity_weight=sc["cf_w"],
                diffusion_strength=sc["sd_s"],
                enable_codeformer=sc["cf"],
                enable_diffusion=sc["sd"],
                warp_strength=0.25,
                vertical_strength=0.10,
                radial_k1=-0.05
            )
            
            # Save rectified, final, and debug outputs to brain directory
            rectified_out = brain_dir / f"id_test_{name}_rectified.png"
            final_out = brain_dir / f"id_test_{name}_final.png"
            debug_out = brain_dir / f"id_test_{name}_debug.png"
            
            save_image(res["rectified"], rectified_out)
            save_image(res["final"], final_out)
            save_image(res["debug_vis"], debug_out)
            
            print(f"[OK] Saved {name} rectified to {rectified_out}")
            print(f"[OK] Saved {name} final to {final_out}")
            print(f"[OK] Saved {name} debug visualization to {debug_out}")
        except Exception as e:
            print(f"[ERROR] Scenario {name} failed: {e}")

if __name__ == "__main__":
    main()
