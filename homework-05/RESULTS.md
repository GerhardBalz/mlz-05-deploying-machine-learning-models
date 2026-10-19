# Homework 5 — Deploying Machine Learning Models

Completed by Gerhard Balz on 19 October 2026 (Europe/Zurich).

Status at this review: exercises complete; results prepared locally; solution not published and course submission not performed. The container stopped cleanly. The retained Codespace is still available.

## Answers

| Question | Observed result | Answer to enter |
| --- | --- | --- |
| Q1 — local uv version | uv 0.10.11 | 0.10.11 |
| Q2 — locked scikit-learn | declared and locked at 1.7.2 | 1.7.2 |
| Q3 — direct inference, first lead | 0.5329944186432507 | 0.533 |
| Q4 — API inference, second lead | 0.769799; conversion true | 0.770 |
| Q5 — declared Python base-image tag | python:3.11.15-slim-bookworm | python:3.11.15-slim-bookworm |
| Q6 — container inference, second lead | 0.769799; conversion true | 0.770 |

Q1 discrepancy: the official homework calls this an ungraded environment check, while the portal was observed to label it one point. The actual local version is recorded without assuming a grading rule. Q3, Q4 and Q6 require numeric probabilities rounded to three decimal places; the stated absolute tolerance is 0.005.

## Sources and environment

- [Course portal](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05)
- [Module lessons](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/05-deployment/README.md)
- [Updated FastAPI/uv workshop](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/05-deployment/workshop/README.md)
- [Pinned 2026 homework requirements](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/33ad5e9ad4d56fcfdd8ff162eca75b7628c8d1ed/cohorts/2026/homework/05-deployment/homework.md)
- [Working repository](https://github.com/GerhardBalz/mlz-05-deploying-machine-learning-models)
- [Setup and Q1–Q2 evidence](https://github.com/GerhardBalz/mlz-01-introduction-to-machine-learning/issues/1#issuecomment-6265444684)
- [Shared Zoomcamp workflow](https://github.com/GerhardBalz/ai-workspace-governance/blob/main/docs/runbooks/zoomcamp-homework-workflow.md)

The supplied release files were obtained from DataTalksClub/machine-learning-zoomcamp at revision `33ad5e9ad4d56fcfdd8ff162eca75b7628c8d1ed`, directory `cohorts/2026/homework/05-deployment`. The exact raw source URL is retained in [SOURCE.txt](SOURCE.txt). The original filenames are preserved.

The isolated environment is `homework-05/.venv`, Python 3.11.15, installed with uv 0.10.11 using `uv sync --locked`. Required imports passed during setup. The selected workspace interpreter was independently observed at `homework-05/.venv/bin/python`.

Declared dependencies: FastAPI 0.119.1, NumPy 2.3.3, scikit-learn 1.7.2 and Uvicorn 0.38.0. The supplied lockfile was preserved. Its learner-reported Git blob hash was `3bc62539700fbaa7a358a6b098de37344e406686`, matching the pinned source.

The frozen model is a fitted DictVectorizer followed by LogisticRegression with solver liblinear and C=1.0. The saved pipeline displays max_iter=1000 and random_state=42. No retraining or artifact replacement was performed.

Model metadata identifies `course_lead_scoring_2026.csv` (5,000 rows), dataset SHA-256 `2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9`. The training CSV is not required for this frozen-model inference exercise and was not downloaded for Homework 5. Missing categories use NA; missing numeric values use the recorded training-data medians.

## Q3 — load the fitted pipeline

The learner-created [q3.py](q3.py) loads `pipeline.bin` with pickle, uses the exact first lead from the homework, and evaluates:

```python
probability = pipeline.predict_proba([datapoint])[0, 1]
```

The single-item list is a one-row batch. Classes are ordered [0, 1], so column 1 is conversion.

Learner-run command, from this directory:

```bash
.venv/bin/python q3.py
```

Reported output:

```text
Pipeline(steps=[('dictvectorizer', DictVectorizer()),
                ('logisticregression',
                 LogisticRegression(max_iter=1000, random_state=42,
                                    solver='liblinear'))])
Classes: [0 1]
Conversion probability: 0.5329944186432507
Q3 answer: 0.533
```

## Q4 — serve the supplied API

Learner-run server command:

```bash
uv run --locked uvicorn predict:app --host 0.0.0.0 --port 9696
```

In a second terminal, the learner sent the exact second lead:

```bash
curl -s http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'
```

Reported response:

```json
{"conversion_probability":0.769799,"conversion":true}
```

The API rounds its probability field to six decimals. The homework answer is rounded to three decimals: 0.770. The conversion decision uses probability >= 0.5.

The server logged HTTP 200 for POST /predict. Browser requests to / and /favicon.ico returned 404 because those routes are not defined. They were not prediction failures.

## Q5 — inspect the canonical container definition

Learner inspected the FROM declaration:

```dockerfile
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
```

The tag is the Q5 answer. The digest pins the exact image content. The Dockerfile also pins uv 0.10.11 by digest, installs using `uv sync --locked --no-dev`, and copies the supplied application and frozen model.

## Q6 — run the same inference in Docker

Learner-run build:

```bash
docker build -t zoomcamp-model:2026-hw5 .
```

Reported build completed all 15 steps successfully. Exported image manifest list: `sha256:6671512e26f4b7ca9a119c7206a48437e95727944ae7b9bd712b3b11c3b1686c`. This identifies the learner's local build; no registry push was performed.

The direct Uvicorn server was stopped cleanly to release port 9696. The learner then ran:

```bash
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
```

Container startup completed successfully. In the second terminal:

```bash
.venv/bin/python q6_test.py
```

Reported client output:

```text
{'conversion_probability': 0.769799, 'conversion': True}
```

The supplied client checks that the response has exactly the probability and conversion fields, the probability is in [0, 1], and conversion is boolean. It exited without a reported assertion error. The returned probability matches Q4 exactly at the API's six-decimal precision. Q6 answer: 0.770.

Learner-run health check:

```bash
curl -s http://localhost:9696/health
```

Reported response:

```json
{"status":"ok","model_sha256":"1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1","feature_defaults_sha256":"1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"}
```

Both checksums match the pinned official metadata and the setup checks. The container was stopped with Ctrl+C after these checks; shutdown-complete logs and a returned shell prompt were independently observed in the editor. The --rm option makes this a disposable container; the image is retained.

Use the Homework 5 tag `zoomcamp-model:2026-hw5`. The supplied shared-release README includes an HW10 example tag, which was not used for this exercise.

## Review and evidence boundaries

All execution results above are learner-run, pasted outputs, not assistant calculation replays. Q1–Q2 setup evidence is carried forward from the linked verified handover. Q3–Q6 were run by the learner on 19 October 2026.

During this review, the saved Q3 code was independently read and checked against the official first lead and class indexing. The 13 supplied text files (.python-version, Dockerfile, README.md, feature_defaults.json, homework.md, model.py, model_metadata.json, predict.py, pyproject.toml, q6_test.py, smoke_test.py, train.py, uv.lock) were read from the editor and compared with the pinned official revision after normalizing clipboard line endings. They matched. This is a text-content comparison, not a new byte-level audit of every local file. The binary model was not reloaded or rehashed by the assistant; its identity is supported by the learner's setup checksum and container health response.

SOURCE.txt points to the pinned directory. The root starter README and .gitignore were preserved. The original .vscode/settings.json remains local and contains an environment-manager preference; it is outside the proposed solution publication set. The ignored .venv and generated __pycache__ remain intact.

The source train.py is retained for provenance. Its original dataset-relative default path assumes the upstream directory layout; it was not executed or adapted for this inference-only homework. No smoke-test replay, model inference replay, dependency installation, lock regeneration, Docker rebuild, course-form update or social sharing was performed during review.

Public main was independently read during review and still points to starter commit `d832ac151f0667920a282192f8145514075f86bd`. The local results and solution files are uncommitted. The Homework 5 portal was last verified at the preceding closeout as NOT SUBMITTED / Not saved yet, with all fields blank and deadline 27 October 2026 at 00:00 account time; it was not rechecked or changed during this file review.

## Proposed publication set — requires a separate decision

- This RESULTS.md record and q3.py.
- SOURCE.txt and all 14 supplied release files, including pipeline.bin and uv.lock.
- Preserve the existing root README and .gitignore.
- Retain .vscode/settings.json, .venv and __pycache__ locally; exclude them from the solution commit.

Publication, any merge, external deployment, course submission and social sharing remain separate decisions. No such action is authorized by this record.
