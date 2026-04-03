# X-Ray Baggage Object Detection — Classical CV

**Author:** Henri Boisson  
**Score:** 0.28 mean IoU (#1 on leaderboard)

## Approach

Two-stage classical detection pipeline:

1. **Multi-method region proposals**: Combines multi-level thresholding (OTSU + 7 fixed + adaptive), Canny edge detection (6 configs), morphological top-hat/black-hat (3 scales), and MSER blob detection. All proposals fused via NMS. Achieves 99.7% recall on validation.

2. **Rich feature extraction** (~77 dims per crop): Intensity statistics, 16-bin histogram, Sobel gradient + orientation histogram, shape/contour metrics, Hu moments, LBP texture, spatial quadrant features, crop size.

3. **XGBoost 7-class classifier** (6 objects + background): Trained on ~60k samples after 2 rounds of hard negative mining. Inverse-frequency sample weighting for class balance.

4. **Post-processing**: Confidence gate (>0.5), per-class size filter, per-class NMS, cross-class NMS, max 3 detections/image.

## Files

- `submission.csv` — Final predictions on test set
- `henri_boisson.ipynb` — Complete pipeline (train + test)
- `requirements.txt` — Dependencies
- `models/pipeline.pkl` — Trained XGBoost model + scaler + class size priors
- `report.docx` — 1-page report

## How to Run

1. Open `henri_boisson.ipynb` in Google Colab
2. Run Cell 1 (installs dependencies), then **restart runtime**
3. Run all remaining cells top-to-bottom
4. `submission.csv` is generated automatically

## Dependencies

See `requirements.txt`. Key: `xgboost`, `opencv-contrib-python`, `scikit-image`, `scikit-learn`.
