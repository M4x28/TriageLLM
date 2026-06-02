"""Generate triage_seed.jsonl: authored triage-behavior examples.

Phase 2 under-triage fix. This is the primary carrier of three signals the
MIETIC-dominated corpus lacks:
  1. pediatric danger sign -> Action: REFER NOW (the failure_to_escalate gap),
  2. genuinely non-urgent child  -> Action: HOME CARE / ROUTINE (anti over-triage),
  3. non-clinical / out-of-domain -> Action: OUT-OF-SCOPE, no forced ESI.

Same safety pattern as generate_identity_seed.py: the ANSWER is template-fixed
(clinically reviewed, never LLM-generated) and the LLM only paraphrases the
QUESTION to add surface diversity. Answers are Action-first, anchored to WHO
IMCI/ETAT, contain no infant (<3mo) dosing and no drug dose without deferring to
a clinician.

Runs ONCE on the server; output committed for reproducibility.

Usage (server):
  python 4_data_augment/generate_triage_seed.py \
    --model Qwen/Qwen3-32B --variants 8 --out 4_data_augment/triage_seed.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from common import VLLMBatchClient, VLLMConfig, setup_logging, write_jsonl

# Single source of truth: the same system prompt the model is trained/served with.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "3_qa_rewrite"))
from prompts import SYSTEM_PROMPT  # noqa: E402

PERSONA_DESC = {
    "caregiver_query": "family member or patient (non-clinician) asking the chatbot",
    "clinician_handoff": "nurse or doctor handing off a case for decision support",
    "field_worker_query": "triage staff in a low-resource clinic",
}

# Authored seeds. `framework`/`action` populate metadata so step 4 oversampling
# and the Action-first guard treat them like derived clinical records.
SEEDS: list[dict] = [
    # ───── pediatric danger sign -> REFER NOW (IMCI/ETAT) ─────
    {
        "persona": "caregiver_query", "category": "danger_sign",
        "framework": "IMCI_ETAT", "action": "REFER NOW",
        "q": "My 3-year-old just had a fit, his whole body shook and his eyes rolled back, and now he is very sleepy.",
        "a": "Action: REFER NOW\n\nA convulsion (the shaking with eyes rolling back) and the drowsiness afterwards are WHO IMCI/ETAT danger signs in a young child. Take him to the nearest health facility immediately, even if he now seems to be recovering. On the way, lay him on his side, do not put anything in his mouth, and keep him warm. The facility may check his blood sugar, test and treat for malaria or other infection, and give emergency care if needed.",
    },
    {
        "persona": "caregiver_query", "category": "danger_sign",
        "framework": "IMCI_ETAT", "action": "REFER NOW",
        "q": "My baby vomits everything I give her and has refused to breastfeed since this morning.",
        "a": "Action: REFER NOW\n\nNot being able to feed or drink, and vomiting everything, are WHO IMCI danger signs. A young child can become dehydrated and weak very quickly. Take her to the nearest health facility now. You can offer small sips on the way if she will take them, but do not delay going. The facility can rehydrate her and find the cause of the vomiting.",
    },
    {
        "persona": "field_worker_query", "category": "danger_sign",
        "framework": "IMCI_ETAT", "action": "REFER NOW",
        "q": "Child about 2 years brought in floppy and very hard to wake, fever for 3 days. We have limited supplies here.",
        "a": "Action: REFER NOW\n\nLethargy or being difficult to wake is an ETAT emergency sign. This child needs urgent referral to the nearest facility with inpatient capacity. While you arrange transport, follow ETAT: check and support airway and breathing, check blood glucose and treat hypoglycaemia if you can, keep the child warm, and begin malaria assessment. Do not wait for test results before referring.",
    },
    {
        "persona": "clinician_handoff", "category": "danger_sign",
        "framework": "IMCI_ETAT", "action": "REFER NOW",
        "q": "4-year-old, barking cough, stridor at rest, chest indrawing. Disposition?",
        "a": "Action: REFER NOW\n\nStridor in a calm child together with chest indrawing is an ETAT emergency sign of severe respiratory distress. Refer now to a facility able to manage the airway. Give oxygen if available and keep the child calm and upright during transport. The receiving facility may provide nebulised treatment, steroids, or airway support as indicated.",
    },
    {
        "persona": "caregiver_query", "category": "danger_sign",
        "framework": "IMCI_ETAT", "action": "REFER NOW",
        "q": "My child has had watery diarrhea all day, his eyes look sunken and when I pinch the skin it stays up.",
        "a": "Action: REFER NOW\n\nSunken eyes and skin that stays pinched are signs of severe dehydration, a WHO IMCI danger sign. Take your child to the nearest health facility now. Keep giving oral rehydration solution or clean fluids in small, frequent sips on the way. The facility may give fluids through a drip and treat the cause of the diarrhea.",
    },
    {
        "persona": "field_worker_query", "category": "danger_sign",
        "framework": "IMCI_ETAT", "action": "REFER NOW",
        "q": "Mother says her child 'shook all over for a few minutes and then was confused.' No fit now. Can we manage locally?",
        "a": "Action: REFER NOW\n\nA described episode of shaking all over followed by confusion should be treated as a convulsion with a post-ictal state, an ETAT/IMCI danger sign, even though the child is not fitting now. Refer to the nearest facility for assessment. Do not delay because the episode has stopped or because the history is uncertain. The facility can check blood sugar, screen for malaria and meningitis, and observe.",
    },
    # ───── genuinely non-urgent child -> HOME CARE / ROUTINE (anti over-triage) ─────
    {
        "persona": "caregiver_query", "category": "non_urgent",
        "framework": "IMCI_ETAT", "action": "HOME CARE + RETURN ADVICE",
        "q": "My 4-year-old has had a mild fever since yesterday but she is drinking, eating, and playing normally.",
        "a": "Action: HOME CARE + RETURN ADVICE\n\nA child with a mild fever who is alert, drinking, eating, and playing has no WHO IMCI danger signs right now. You can care for her at home: give plenty of fluids, keep her comfortable, and use weight-based paracetamol only if a health worker has advised it. Return to a health facility immediately if she develops any danger sign: stops drinking or feeding, vomits everything, has a convulsion, breathes fast or with difficulty, becomes drowsy or hard to wake, or if the fever lasts more than 2-3 days.",
    },
    {
        "persona": "caregiver_query", "category": "non_urgent",
        "framework": "IMCI_ETAT", "action": "HOME CARE + RETURN ADVICE",
        "q": "My toddler has a runny nose and a slight cough but is active and feeding well.",
        "a": "Action: HOME CARE + RETURN ADVICE\n\nA runny nose and a mild cough in a child who is active and feeding well is usually a simple cold with no danger signs. Care for her at home with fluids, normal feeding, and clearing the nose so she breathes more easily. Return urgently if you see fast or difficult breathing, chest indrawing, the child stops feeding or drinking, becomes drowsy, or has a convulsion.",
    },
    {
        "persona": "field_worker_query", "category": "non_urgent",
        "framework": "IMCI_ETAT", "action": "ROUTINE FOLLOW-UP",
        "q": "5-year-old, mild sore throat, no fever, eating normally, breathing normal. Should I refer?",
        "a": "Action: ROUTINE FOLLOW-UP\n\nWith no fever, normal breathing, and normal feeding, there are no IMCI/ETAT danger signs and no need for urgent referral. Manage the symptoms and arrange routine follow-up. Advise the caregiver to return promptly if the child develops fast breathing, stops eating or drinking, becomes drowsy, or develops a high fever.",
    },
    # ───── non-clinical / out-of-domain -> OUT-OF-SCOPE (no forced ESI) ─────
    {
        "persona": "caregiver_query", "category": "out_of_domain",
        "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
        "q": "Can you write an email to my child's school explaining she will be absent?",
        "a": "Action: OUT-OF-SCOPE\n\nThis is not a clinical triage question, so there is no triage assessment to give. I am a triage decision-support assistant focused on recognizing danger signs and deciding the right level of care. If your child is unwell, tell me her symptoms and I can help you judge what care she may need.",
    },
    {
        "persona": "caregiver_query", "category": "out_of_domain",
        "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
        "q": "Translate 'good morning, how are you' into French for me.",
        "a": "Action: OUT-OF-SCOPE\n\nThat is a general request, not a clinical triage question, so I will not assign a triage level. I am built to assess danger signs and decide the right level of care for a patient. If you have a patient or a symptom to triage, describe it and I will help.",
    },
    {
        "persona": "field_worker_query", "category": "out_of_domain",
        "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
        "q": "What is 25 times 17?",
        "a": "Action: OUT-OF-SCOPE\n\nThis is not a clinical triage question, so there is no triage disposition here. I focus on recognizing danger signs and recommending the right level of care. If you give me a patient's symptoms, I will help you triage them.",
    },
    {
        "persona": "clinician_handoff", "category": "out_of_domain",
        "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
        "q": "Draft a stock order for amoxicillin for our clinic pharmacy.",
        "a": "Action: OUT-OF-SCOPE\n\nStock ordering is an administrative task, not a clinical triage decision, so I will not give a triage level or a prescription. I can help you triage a specific patient and cite WHO treatment criteria for that case, but procurement and pharmacy orders are outside what I do.",
    },
]


PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {"type": "string"}}},
    "required": ["variants"],
}


def build_paraphrase_prompt(persona: str, base_q: str, n_variants: int) -> str:
    return (
        "Generate exactly {n} short paraphrases of the following user "
        "message. Keep the same intent, the same clinical facts, and the same "
        "persona ({persona}: {desc}). Vary phrasing, vocabulary, and tone. Each "
        "paraphrase must be a self-contained user-facing utterance, between 5 "
        "and 40 words. Do NOT answer it. Do NOT add explanations. Do NOT change "
        "or drop any symptom. Return JSON with key 'variants' containing the "
        "array of paraphrases.\n\n"
        'Original message: "{q}"\n\nOutput JSON only.'.format(
            n=n_variants, persona=persona, desc=PERSONA_DESC[persona], q=base_q,
        )
    )


def main() -> int:
    log = setup_logging("triagellm.augment.triage_seed")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--variants", type=int, default=8,
                    help="Paraphrase variants per seed (final ~13 × (1 + variants))")
    ap.add_argument("--out", default=str(Path(__file__).parent / "triage_seed.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    client = VLLMBatchClient(VLLMConfig(model_name=args.model))
    prompts = [build_paraphrase_prompt(s["persona"], s["q"], args.variants) for s in SEEDS]
    log.info("running %d paraphrase batches × %d variants", len(prompts), args.variants)
    results = client.batch_extract(prompts, PARAPHRASE_SCHEMA,
                                   max_tokens=args.max_tokens, temperature=0.7)

    records: list[dict] = []
    idx = 0
    for seed, res in zip(SEEDS, results):
        questions = [seed["q"]]
        for v in (res or {}).get("variants") or []:
            v = (v or "").strip().strip('"').strip()
            if v and v not in questions and 5 <= len(v.split()) <= 80:
                questions.append(v)
        for q in questions:
            records.append({
                "id": f"triage_{idx:04d}_{seed['category']}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": seed["a"]},
                ],
                "metadata": {
                    "source_id": f"triage_seed_{seed['category']}_{seed['persona']}",
                    "source": "triage_seed",
                    "doc_type": "triage_behavior",
                    "style": seed["persona"],
                    "category": seed["category"],
                    "triage_framework": seed["framework"],
                    "action": seed["action"],
                    "prompt_version": "v2",
                },
            })
            idx += 1

    n = write_jsonl(Path(args.out), records)
    log.info("wrote %d triage-seed records → %s", n, args.out)
    for k, v in Counter(r["metadata"]["category"] for r in records).most_common():
        log.info("  %-14s %d", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
