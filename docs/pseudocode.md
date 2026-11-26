# Pseudocode of implemented modules

Written in the template from the Phase-1 Presentation 3 guidelines (MODULE, PURPOSE, INPUT, OUTPUT, BEGIN ... END). The same text is on the pseudocode slides.

## F01 Sign In
```
MODULE:  F01 Sign In
PURPOSE: let only registered, active staff into the system
INPUT:   username, password
OUTPUT:  session with user and role, or an error
BEGIN
    Receive username and password
    user ← active user with this username
    IF user = NULL OR PBKDF2(password, salt) ≠ hash THEN
       log failed_login
       RETURN 'Invalid username or password.'
    ELSE
       store user and role in the session; log login
       RETURN sidebar of pages allowed for the role
    END IF
END
```

## F06 Data Management
```
MODULE:  F06 Data Management
PURPOSE: validate and clean raw student data
INPUT:   raw CSV file (or bundled dataset)
OUTPUT:  cleaned dataset and cleaning report
BEGIN
    Receive CSV file and read it
    IF required columns missing OR unknown source THEN
       RETURN 'Schema problems ...' (cleaning blocked)
    ELSE
       standardise text; remove duplicate rows
       set fields a source never measures to NULL
       clip to valid ranges; flag 3×IQR outliers
       fill missing values (median / mode per source)
       drop rows without grade; A,B→Low, C→Medium, D→High
       log clean_dataset; RETURN report and CSV
    END IF
END
```

## Model Training Pipeline
```
MODULE:  Model Training Pipeline
PURPOSE: train, compare and save the risk model (results shown in F08)
INPUT:   raw dataset file
OUTPUT:  best model, results files, registered versions
BEGIN
    Receive dataset path
    IF file missing OR schema invalid THEN
       RETURN error and stop
    ELSE
       clean data (as in F06); X ← features without outcome columns
       split 80/20 stratified on source and risk
       FOR m IN {LR, RF, GB}: tune with 5-fold CV
          record macro-F1, 95% CI and test scores
       best ← highest mean CV macro-F1 (Random Forest)
       per-source, ablation and calibration checks
       SHAP results; fairness audit (F10)
       save best (Joblib); register versions
    END IF
END
```

## F03 Risk Prediction
```
MODULE:  F03 Risk Prediction
PURPOSE: predict one student's risk and explain it
INPUT:   record type and its fields
OUTPUT:  risk, confidence, reasons, interventions
BEGIN
    Receive form values (form limits block impossible values)
    IF trained model not found THEN
       RETURN 'Run python run_pipeline.py'
    ELSE
       clip values; blank fields not measured for the type
       p ← model.predict_proba(x); risk ← class with max p
       φ ← SHAP values for High: raising and lowering factors
       text ← plain-English explanation
       actions ← intervention for each top raising factor
       save student, features, prediction, 10 factors; log
       RETURN results and HTML report
    END IF
END
```

## F04 Batch Prediction
```
MODULE:  F04 Batch Prediction
PURPOSE: predict risk for a whole class from one file
INPUT:   CSV file of students
OUTPUT:  results table, CSV download, saved batch
BEGIN
    Receive CSV file
    IF 'source' column missing OR unknown source THEN
       RETURN error (nothing predicted or saved)
    ELSE
       FOR each row: p ← model.predict_proba(row)
          risk, confidence, top-3 risk-raising factors
       show counts, chart and table; offer CSV
       IF user clicks Save THEN
          create batch; save each prediction; log
       END IF
       RETURN results
    END IF
END
```

## F10 Fairness Audit
```
MODULE:  F10 Fairness Audit
PURPOSE: check that High-risk students are found equally in every group
INPUT:   test predictions; gender, income group, age group
OUTPUT:  recall per group, gap with 95% CI, methods compared
BEGIN
    Receive predictions and groups; flag ← (prediction = High)
    remove students whose group is 'Unknown'
    IF fewer than 2 groups have High-risk students THEN
       RETURN gap = not defined
    ELSE
       FOR each group: recall, missed, flagged, macro-F1
       gap ← max − min recall; 95% CI (500 bootstraps)
       FOR gender, income: Reweighing, ExpGradient,
          Threshold Optimizer; compare with baseline
       save results; RETURN to the Fairness Audit page
    END IF
END
```
