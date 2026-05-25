import os
from pathlib import Path
import gradio as gr
import gradio_client.utils as gradio_client_utils
from fastapi.templating import Jinja2Templates
from PIL import Image

from scripts.pipeline import SelfieRectifier
from scripts.utils import save_image

# Workaround for a gradio_client schema parsing bug when boolean JSON schema values are encountered.
# Gradio may generate a schema where `additionalProperties` is a bool, causing `get_type(schema)`
# to crash inside gradio_client.utils. This patch ensures boolean JSON schema values are handled.
_original_gradio_client_get_type = gradio_client_utils.get_type

def _safe_gradio_client_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    if not isinstance(schema, dict):
        return {}
    return _original_gradio_client_get_type(schema)

gradio_client_utils.get_type = _safe_gradio_client_get_type

# Workaround for a FastAPI/Gradio compatibility mismatch in the local environment.
# Older Gradio code uses the legacy TemplateResponse(template_name, context) signature,
# while the installed FastAPI version expects TemplateResponse(request, template_name, context).
_original_jinja2_template_response = Jinja2Templates.TemplateResponse

def _legacy_compatible_template_response(self, *args, **kwargs):
    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
        template_name, context = args[0], args[1]
        request = context.pop("request", None)
        if request is None:
            raise ValueError("Missing request object when calling TemplateResponse")
        return _original_jinja2_template_response(self, request, template_name, context, **kwargs)
    return _original_jinja2_template_response(self, *args, **kwargs)

Jinja2Templates.TemplateResponse = _legacy_compatible_template_response

def create_ui():
    print("[INFO] Initializing Selfie Rectification Pipeline models...")
    try:
        rectifier = SelfieRectifier()
        print("[INFO] Models initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize pipeline: {str(e)}")
        raise

    def process(
        image, 
        enable_cf, 
        cf_weight, 
        enable_sd, 
        sd_strength, 
        sd_guidance,
        enable_de,
        de_blend_weight,
        de_sharpen_strength,
        de_eye_boost,
        de_hair,
        face_upsample,
        warp_strength, 
        vertical_strength, 
        radial_k1
    ):
        if image is None:
            return [None] * 11 + [None, None]
        try:
            temp_dir = Path("outputs") / "gradio"
            temp_dir.mkdir(parents=True, exist_ok=True)
            input_path = temp_dir / "input.png"
            save_image(image.convert("RGB"), input_path)

            print("[INFO] Running image through rectification pipeline...")
            res = rectifier.process_with_intermediates(
                input_path,
                fidelity_weight=cf_weight,
                diffusion_strength=sd_strength,
                enable_codeformer=enable_cf,
                enable_diffusion=enable_sd,
                shift=1.2,
                warp_strength=warp_strength,
                vertical_strength=vertical_strength,
                radial_k1=radial_k1,
                enable_detail_enhancer=enable_de,
                detail_blend_weight=de_blend_weight,
                sharpen_strength=de_sharpen_strength,
                eye_boost=de_eye_boost,
                hair_preservation=de_hair,
                diffusion_guidance=sd_guidance,
                face_upsample=face_upsample,
            )
            print("[INFO] Inference complete.")
            
            # Compute difference heatmap
            import numpy as np
            import cv2
            try:
                arr_orig = np.array(image.convert("L"), dtype=np.float32)
                arr_final = np.array(res["final"].convert("L"), dtype=np.float32)
                diff = np.abs(arr_orig - arr_final)
                diff_norm = np.clip(diff * 4.0, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                diff_heatmap = Image.fromarray(heatmap_rgb)
            except Exception as e_heat:
                print(f"[WARN] Failed to compute difference heatmap: {e_heat}")
                diff_heatmap = image
                
            metrics = res["metrics"]
            
            return (
                res["landmarks"],
                res["depth"],
                res["rectified"],
                res["codeformer"],
                res["final"],
                res["debug_vis"],
                diff_heatmap,
                f"{metrics['psnr']:.2f} dB",
                f"{metrics['ssim']:.4f}",
                f"{metrics['lpips']:.4f}",
                f"{metrics['identity_similarity']:.4f}",
                str(res["final_path"]),
                str(res["report_path"]) if res["report_path"] else None
            )
        except Exception as e:
            print(f"[ERROR] Inference failed: {str(e)}")
            raise gr.Error(f"Inference failed: {str(e)}")

    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="slate")) as demo:
        gr.HTML("""
        <div style="text-align: center; padding: 25px 0;">
            <h1 style="font-size: 2.6rem; margin-bottom: 8px;">Geometry-Aware Selfie Rectification</h1>
            <p class="subtitle">Diffeomorphic RK4-Optimized 3D Camera Reprojection & Restored Face Synthesis</p>
            <div style="height: 1px; background: linear-gradient(to right, transparent, #30363d, transparent); width: 80%; margin: 15px auto;"></div>
        </div>
        """)

        with gr.Tabs():
            with gr.TabItem("Interactive Rectification"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(type="pil", label="Upload Original Selfie")
                        
                        with gr.Group():
                            gr.Markdown("### ⚙️ Identity & Refinement Controls")
                            warp_strength = gr.Slider(
                                label="Warp Strength (Subtle Correction)", 
                                minimum=0.0, 
                                maximum=1.0, 
                                value=0.25, 
                                step=0.05
                            )
                            vertical_strength = gr.Slider(
                                label="Vertical Scaling Constraint (Bypassed Internally to Enforce 0)", 
                                minimum=0.0, 
                                maximum=1.0, 
                                value=0.10, 
                                step=0.05,
                                interactive=False
                            )
                            radial_k1 = gr.Slider(
                                label="Radial Lens Distortion (K1 Barrel Compensation)", 
                                minimum=-0.30, 
                                maximum=0.30, 
                                value=-0.05, 
                                step=0.01
                            )
                            enable_cf = gr.Checkbox(label="Enable CodeFormer Face Restoration", value=True)
                            cf_weight = gr.Slider(
                                label="CodeFormer Fidelity (Higher = Keeps Identity)", 
                                minimum=0.1, 
                                maximum=1.0, 
                                value=0.95, 
                                step=0.05
                            )
                            face_upsample = gr.Checkbox(label="Enable CodeFormer Face Upsampling (Lightweight SR)", value=False)
                            enable_sd = gr.Checkbox(label="Enable Stable Diffusion Refinement", value=False)
                            sd_strength = gr.Slider(
                                label="SD Denoising Strength (Higher = Changes Identity)", 
                                minimum=0.05, 
                                maximum=0.20, 
                                value=0.10, 
                                step=0.01
                            )
                            sd_guidance = gr.Slider(
                                label="SD Guidance Scale (Lower CFG = Structure Preserving)",
                                minimum=1.0,
                                maximum=10.0,
                                value=3.0,
                                step=0.5
                            )
                            
                            gr.Markdown("### 🛡️ Detail Preservation & Sharpening")
                            enable_de = gr.Checkbox(label="Enable Detail Preservation Pass", value=True)
                            de_blend_weight = gr.Slider(
                                label="Detail Blend Weight (Retain original pores/texture)",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.6,
                                step=0.05
                            )
                            de_sharpen_strength = gr.Slider(
                                label="Edge-Aware Sharpening Strength",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,
                                step=0.05
                            )
                            de_eye_boost = gr.Slider(
                                label="Localized Eye Sharpening Boost",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.4,
                                step=0.05
                            )
                            de_hair = gr.Slider(
                                label="Hair Detail Preservation",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.05
                            )
                            
                        rectify_btn = gr.Button("Run Perspective Rectification", variant="primary", elem_classes=["rectify-btn"])
                        
                    with gr.Column(scale=1):
                        output_img = gr.Image(type="pil", label="Rectified Final Output", interactive=False)
                        download_file = gr.File(label="Download Rectified Selfie", interactive=False)
                        
                        gr.Markdown("### 📊 Real-Time Benchmark Scores")
                        with gr.Row():
                            psnr_txt = gr.Textbox(label="PSNR (Face ROI)", interactive=False)
                            ssim_txt = gr.Textbox(label="SSIM (Face ROI)", interactive=False)
                        with gr.Row():
                            lpips_txt = gr.Textbox(label="LPIPS (Perceptual)", interactive=False)
                            id_txt = gr.Textbox(label="ArcFace Score (Identity)", interactive=False)
                        download_report = gr.File(label="Download Interactive HTML Research Report", interactive=False)

                gr.HTML("""
                <div style="margin-top: 30px; margin-bottom: 15px;">
                    <h3 style="color: #58a6ff; border-left: 4px solid #1f6feb; padding-left: 10px;">Pipeline Step-by-Step Visualization</h3>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Step 1: Face Mesh (Landmarks)**")
                        stage1_img = gr.Image(type="pil", label="Landmarks detected", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("**Step 2: Estimated Depth**")
                        stage2_img = gr.Image(type="pil", label="Depth Anything V2 Map", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("**Step 3: Perspective Corrected**")
                        stage3_img = gr.Image(type="pil", label="Focal Length Warped", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("**Step 4: CodeFormer Restored**")
                        stage4_img = gr.Image(type="pil", label="Face Detail Enhanced", interactive=False)

                gr.HTML("""
                <div style="margin-top: 30px; margin-bottom: 15px;">
                    <h3 style="color: #ff7b72; border-left: 4px solid #da3633; padding-left: 10px;">Geometry & Distortion Debug Panels</h3>
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Geometry & Distortion Diagnostic Visualizations**")
                        debug_vis_panel = gr.Image(type="pil", label="ROI Mask, Warp Mesh, Deformation Vectors, Landmarks Comparison", interactive=False)
                    with gr.Column(scale=1):
                        gr.Markdown("**Geometry Shift Heatmap (Red = Max Displacement)**")
                        diff_heatmap_img = gr.Image(type="pil", label="Difference Heatmap Colormap", interactive=False)

                rectify_btn.click(
                    fn=process,
                    inputs=[
                        input_img,
                        enable_cf,
                        cf_weight,
                        enable_sd,
                        sd_strength,
                        sd_guidance,
                        enable_de,
                        de_blend_weight,
                        de_sharpen_strength,
                        de_eye_boost,
                        de_hair,
                        face_upsample,
                        warp_strength,
                        vertical_strength,
                        radial_k1
                    ],
                    outputs=[
                        stage1_img,
                        stage2_img,
                        stage3_img,
                        stage4_img,
                        output_img,
                        debug_vis_panel,
                        diff_heatmap_img,
                        psnr_txt,
                        ssim_txt,
                        lpips_txt,
                        id_txt,
                        download_file,
                        download_report
                    ]
                )

            with gr.TabItem("Research Overview & Math Presentation"):
                gr.HTML("""
                <div style="padding: 20px; line-height: 1.6;">
                    <h2 style="color: #58a6ff; font-size: 1.8rem; margin-top: 0; margin-bottom: 20px; border-bottom: 1px solid #30363d; padding-bottom: 10px;">
                        🔬 Scientific Architecture & Math Formulation
                    </h2>
                    
                    <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px;">
                        <h3 style="color: #ff7b72; margin-top: 0; font-size: 1.2rem;">1. 3D Camera Projection Model</h3>
                        <p>Selfie distortion is a close-range perspective scaling artifact. We reconstruct a metric 3D mesh by scaling normalized coordinate depth from MediaPipe face landmarks to real-world dimensions (assuming a 15cm face width):</p>
                        <pre style="background: #0d1117; padding: 10px; border-radius: 4px; overflow-x: auto; color: #58a6ff;">
Z_i = Z_c + z_raw_i * S
where S = 0.15 / norm_width</pre>
                        <p>We re-project the vertices using a virtual portrait camera setback of <em>dz = 1.2m</em>, simulating a DSLR portrait geometry (50mm focal length):</p>
                        <pre style="background: #0d1117; padding: 10px; border-radius: 4px; overflow-x: auto; color: #58a6ff;">
u_proj = c_x + ( (Z_c + dz) / Z_c ) * ( Z_i / (Z_i + dz) ) * (u_i - c_x)
v_proj = c_y + ( (Z_c + dz) / Z_c ) * ( Z_i / (Z_i + dz) ) * (v_i - c_y)</pre>
                    </div>

                    <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px;">
                        <h3 style="color: #ff7b72; margin-top: 0; font-size: 1.2rem;">2. Diffeomorphic RK4 Path Integration</h3>
                        <p>To avoid coordinate mesh folding under strong shifts, we integrate the displacement vector field as a constant velocity field over time <em>t ∈ [0, 1]</em> using a <strong>Runge-Kutta 4th order numerical integrator</strong>:</p>
                        <pre style="background: #0d1117; padding: 10px; border-radius: 4px; overflow-x: auto; color: #58a6ff;">
k1 = v(x_n)
k2 = v(x_n + 0.5 * dt * k1)
k3 = v(x_n + 0.5 * dt * k2)
k4 = v(x_n + dt * k3)
x_(n+1) = x_n + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)</pre>
                        <p>This path integration guarantees a smooth, mathematically clean deformation without folding artifacts.</p>
                    </div>

                    <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px;">
                        <h3 style="color: #ff7b72; margin-top: 0; font-size: 1.2rem;">3. Bounding-Box Constrained Gaussian Falloff</h3>
                        <p>To ensure 100% untouched background and hair, we enforce a localized radial Gaussian falloff weight centered at the nose tip, combined with a hard linear cutoff beyond 60% of the face radius:</p>
                        <pre style="background: #0d1117; padding: 10px; border-radius: 4px; overflow-x: auto; color: #58a6ff;">
weight_i = exp(-r_dist^2 / (2 * sigma^2)) * clip((0.6 - r_dist) / 0.2, 0.0, 1.0)</pre>
                        <p>The warped central face is seamlessly blended back into the original image using Poisson image cloning (<code>cv2.seamlessClone</code>) to ensure perfect lighting and edge transitions.</p>
                    </div>

                    <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px;">
                        <h3 style="color: #ff7b72; margin-top: 0; font-size: 1.2rem;">4. RTX 3050 6GB Memory Management</h3>
                        <p>To enable heavy AI models (Depth Anything V2, CodeFormer, Stable Diffusion img2img) to run under the 6GB VRAM limit:</p>
                        <ul style="padding-left: 20px; margin: 10px 0;">
                            <li><strong>Lazy Loading & GC Collect</strong>: Models are loaded dynamically only when needed, and garbage collected immediately with <code>torch.cuda.empty_cache()</code> right after execution.</li>
                            <li><strong>FP16 Operations</strong>: Stable Diffusion is run in half-precision format with xFormers memory-efficient attention.</li>
                            <li><strong>CPU Offloading</strong>: CPU offloading is enabled for Stable Diffusion components to offload unused parts of the network to system RAM.</li>
                        </ul>
                    </div>
                </div>
                """)

    return demo

def run_app() -> None:
    demo = create_ui()
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except OSError:
        print("[WARN] Port 7860 is occupied. Finding an available open port...")
        demo.launch(server_name="127.0.0.1", share=False)

if __name__ == "__main__":
    run_app()
