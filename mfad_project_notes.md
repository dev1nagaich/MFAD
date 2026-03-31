# MFAD Project Notes

## Environment
- **Virtual environment name**: `mfadenv` (not myenv)
- **Activation**: `conda activate mfadenv`

## Known Issues to Fix

1. **Biological agent**: `'FaceMeshCompat' object has no attribute 'close'`
   - Need to add `close()` method to FaceMeshCompat class
   - File: agents/biological_plausibility_agent.py

2. **MediaPipe warning still in preprocessing**
   - File: agents/preprocessing_agent.py
   - Needs same compatibility wrapper

3. **Geometry anomaly_score returns None**
   - File: agents/geometry.py
   - Should return numeric value instead of None
   - **MITIGATED**: bayesian.py now skips None/NaN/invalid values

4. **ELA chi2 unrealistic**: 172767168.0
   - File: agents/metadata_agent.py
   - Normal range should be 0-10,000
   - **MITIGATED**: bayesian.py validates range (0-1) and skips invalid scores

5. **Report case_id mismatch**
   - Initial case_id: DFA-2026-TC-1AA68191
   - Report generated: DFA-2026-TC-A8E8D2
   - Final output shows Report: None (should show path)

## Pipeline Status
- Preprocessing: ✓
- Geometry: ⚠ (returns None score)
- Metadata: ⚠ (ELA chi2 unrealistic)
- Frequency: ✓
- Texture: ✓
- Biological: ✗ (FaceMeshCompat.close() missing)
- VLM: ✓
- Fusion: ✓ (but with bad scores)
- Report: ⚠ (case_id mismatch, returns None)
