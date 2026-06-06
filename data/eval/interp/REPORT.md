# Interpretability report — ESI framework leak

Observations only. circuit-tracer is run on the BASE model (transcoders valid); fine-tune evidence (round-3/round-5) is transcoder-free (token-set logprob + logit-lens) and correlational.

## Framework per prompt/model (textual baseline)

| prompt | expected | base | r3 | r5 |
| --- | --- | --- | --- | --- |
| esi_leak_resp | IMCI/ETAT (no ESI) | IMCI | NEITHER | IMCI |
| adult_ed_resp | ESI ok | IMCI | NEITHER | NEITHER |
| benign_peds | no ESI (home care / routine) | IMCI | IMCI | IMCI |
| mietic_vitals | IMCI/ETAT (no ESI), even though ESI is requested | IMCI | NEITHER | MIXED |

**Control gate:** base framework on `esi_leak_resp` = `IMCI`. Base does NOT show the ESI prior -> the leak is born/reinforced by the MIETIC fine-tuning, not present in the base model.

## ESI - IMCI margin (token-set mean logprob; higher = more ESI-leaning)

### anchor: start

| prompt | base | r3 | r5 |
| --- | --- | --- | --- |
| esi_leak_resp | -4.529 | -2.52 | -1.364 |
| adult_ed_resp | -4.24 | -1.78 | -0.788 |
| benign_peds | -4.199 | -2.203 | -1.994 |
| mietic_vitals | -3.222 | -0.982 | -0.535 |

### anchor: post_action

| prompt | base | r3 | r5 |
| --- | --- | --- | --- |
| esi_leak_resp | -4.426 | -2.182 | -3.418 |
| adult_ed_resp | -3.008 | -0.779 | -1.338 |
| benign_peds | -3.605 | -1.596 | -3.182 |
| mietic_vitals | -2.997 | -0.097 | -2.092 |

## Logit-lens differential (peak ESI-IMCI layer, post_action; noisy)

| prompt | base (peak@layer) | r3 | r5 |
| --- | --- | --- | --- |
| esi_leak_resp | +1.44@L0 | +2.08@L13 | +4.26@L27 |
| adult_ed_resp | +1.58@L28 | +3.42@L27 | +4.93@L28 |
| benign_peds | +1.44@L0 | +1.88@L13 | +1.90@L13 |
| mietic_vitals | +2.02@L28 | +4.56@L27 | +3.67@L27 |

## Interpretation (weak)

The analysis suggests that the ESI-label prior remains behaviorally strong in fine-tuned models; circuit tracing on the base model is used only to inspect possible origins of this prior. No safety claim is made; the model still leads `Action: REFER NOW` on danger signs.
