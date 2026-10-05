from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import io

from model.brain_mri.predictor import predict

app = FastAPI(
    title="MedVision AI API",
    description="Brain MRI Tumor Classification API",
    version="1.0.0"
)

# Allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to MedVision AI API",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict_mri(file: UploadFile = File(...)):

    # Validate uploaded file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        # Run the trained PyTorch model
        result = predict(image)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        await file.close()