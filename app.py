import os
import uvicorn
import gradio as gr
from api.main import app

# Minimal Gradio interface mounted at /gradio so HF Gradio SDK health checks pass
with gr.Blocks(title="Sachet AI Backend") as demo:
    gr.Markdown("# 🛡️ Sachet (सचेत) — Scam Call Detection Service")
    gr.Markdown(
        "Backend API and ML service are running.\n\n"
        "- **Web App**: Visit [/](/) for the live interactive scam call analyzer\n"
        "- **API Docs**: Visit [/docs](/docs) for Swagger API documentation\n"
        "- **Health Check**: Visit [/health](/health)"
    )

# Mount Gradio onto the existing FastAPI app at /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
