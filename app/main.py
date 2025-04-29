from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.utils import process_lab_report
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/get-lab-tests")
async def get_lab_tests(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="File must be an image (PNG, JPG, JPEG)")
    
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
            
        result = process_lab_report(contents)
        return JSONResponse(
            status_code=200,
            content=result
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "is_success": False,
                "error": f"Error processing image: {str(e)}"
            }
        )

if __name__ == "__main__":  # Fixed __name__ check
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
