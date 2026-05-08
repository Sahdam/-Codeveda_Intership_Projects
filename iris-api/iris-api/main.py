# main.py
from fastapi         import FastAPI, HTTPException
from pydantic        import BaseModel, Field, field_validator
import joblib
import numpy as np

# ── load model once at startup ───────────────────────────
model  = joblib.load('iris_model_v1.pkl')
LABELS = ['setosa', 'versicolor', 'virginica']

app = FastAPI(
    title       = 'Iris Classifier API',
    description = 'Predicts Iris species from sepal and petal measurements.',
    version     = '1.0.0'
)

# ── request & response schemas ───────────────────────────
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, example=5.1, description='cm')
    sepal_width:  float = Field(..., gt=0, example=3.5, description='cm')
    petal_length: float = Field(..., gt=0, example=1.4, description='cm')
    petal_width:  float = Field(..., gt=0, example=0.2, description='cm')

    @field_validator('sepal_length','sepal_width','petal_length','petal_width')
    @classmethod
    def must_be_realistic(cls, v, info):
        if v > 30:
            raise ValueError(f'{info.field_name} seems unrealistically large')
        return v

class IrisOutput(BaseModel):
    species:     str
    confidence:  float
    probabilities: dict[str, float]

# ── endpoints ────────────────────────────────────────────
@app.get('/', tags=['Health'])
def root():
    return {'status': 'ok', 'model': 'SVM RBF pipeline v1.0'}

@app.get('/health', tags=['Health'])
def health():
    return {'status': 'healthy'}

@app.post('/predict', response_model=IrisOutput, tags=['Prediction'])
def predict(data: IrisInput):
    try:
        X = np.array([[data.sepal_length, data.sepal_width,
                       data.petal_length, data.petal_width]])
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IrisOutput(
        species      = LABELS[pred],
        confidence   = round(float(proba[pred]), 4),
        probabilities= {LABELS[i]: round(float(p), 4)
                        for i, p in enumerate(proba)}
    )