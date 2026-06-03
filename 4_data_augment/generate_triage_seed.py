"""Generate triage_seed.jsonl: authored triage-behavior examples.

Phase 2 under-triage fix, round 2 (targeted at the B4 severe-transcript analysis
in data/eval/behavioral/qwen3-1.7b/severe_transcript_analysis.md). The dominant
real failure was ESI/SATS framework leakage into pediatric under-5 cases (the
model emits "ESI Level 2" instead of IMCI/ETAT REFER NOW) plus conditioning
urgency on fever. So every pediatric danger-sign example here is IMCI/ETAT, leads
with `Action: REFER NOW`, carries NO ESI/SATS code, and states the firm referral
sentence. Balanced hard negatives prevent over-triage; admin-in-clinical-context
examples prevent forced triage labels.

Same safety pattern as generate_identity_seed.py: the ANSWER is template-fixed
(clinically reviewed, never LLM-generated); the LLM only paraphrases the QUESTION
for surface diversity. No infant (<3mo) dosing, no drug dose without deferring to
a clinician.

Runs ONCE on the server; output committed for reproducibility.

Usage (server):
  python 4_data_augment/generate_triage_seed.py \
    --model Qwen/Qwen3-8B --variants 8 --out 4_data_augment/triage_seed.jsonl
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

# Mandatory sentence in every danger-sign (REFER NOW) answer.
_REFER = ("Immediate referral to the nearest health facility is required. "
          "Do not wait at home.")


def _refer(body: str) -> str:
    """Compose a REFER NOW answer: Action line, the firm referral sentence, body.
    No ESI/SATS code (IMCI/ETAT framework)."""
    return f"Action: REFER NOW\n\n{_REFER} {body}"


# Authored seeds. `framework`/`action` populate metadata so step 4 oversampling
# and the Action-first guard treat them like derived clinical records. Pediatric
# danger-sign answers deliberately carry NO ESI/SATS code.
SEEDS: list[dict] = [
    # ===== caregiver resists referral (danger sign present, hold the referral) =====
    {"persona": "caregiver_query", "category": "caregiver_resists",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My child just had a convulsion but the clinic is three hours away and I have no transport. Can I just watch him at home tonight?",
     "a": _refer("A convulsion in a young child is a danger sign and home monitoring is not safe, however far the clinic is. Arrange the fastest transport you can (neighbour, motorbike, community health worker); keep him on his side and do not give anything by mouth if he is drowsy. The distance does not change the need to go now.")},
    {"persona": "caregiver_query", "category": "caregiver_resists",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My baby won't feed and is very weak, but we truly have no money for the hospital. Isn't there something I can do here instead?",
     "a": _refer("Not being able to feed plus weakness are danger signs; there is no safe home substitute. Go to the nearest public facility, where emergency care for children is usually free or reduced; ask the community health worker about transport help. Keep offering small sips on the way, but do not delay going to look for money first.")},
    {"persona": "caregiver_query", "category": "caregiver_resists",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "He's lethargic and hard to wake but it's late. Can we just go first thing tomorrow morning?",
     "a": _refer("A child who is lethargic and hard to wake needs to be seen now, not in the morning; this can worsen overnight. Travel at night if you can, keep him warm and on his side, and watch his breathing. Waiting until morning is the main risk here.")},
    {"persona": "caregiver_query", "category": "caregiver_resists",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "She had bad watery diarrhea and sunken eyes earlier, but she seems a little better now, so maybe we don't need to go?",
     "a": _refer("Signs of severe dehydration mean she still needs urgent care even if she looks briefly better; children can tire and crash after seeming to improve. Keep giving oral rehydration solution in small frequent sips on the way. Do not let a short improvement cancel the referral.")},
    {"persona": "field_worker_query", "category": "caregiver_resists",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Mother of a child with a history of repeated seizures today is refusing referral and wants to take him home. How do I handle this?",
     "a": _refer("Repeated seizures in one day is an emergency and the referral must stand despite refusal. Explain plainly the risk of staying home, document the advice given, offer practical help (transport, accompanying note, what to watch on the way), and involve a senior or the community health worker. Do not downgrade to home care to satisfy the refusal.")},
    {"persona": "caregiver_query", "category": "caregiver_resists",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My son makes a harsh noise breathing in and pulls in his chest, but our last hospital visit went badly and I don't trust them. Can you just help me here?",
     "a": _refer("Stridor with chest indrawing is severe breathing difficulty and needs in-person emergency care; I can't manage it remotely. I understand the bad past experience, but the safe step is still to go now, ideally to a different facility if one is reachable. Keep him calm and upright on the way.")},

    # ===== lay-described convulsion / seizure / fit =====
    {"persona": "caregiver_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My little girl suddenly went stiff and her arms and legs were jerking for a couple of minutes.",
     "a": _refer("Going stiff with jerking arms and legs is a convulsion, a danger sign in a young child. Lay her on her side, do not put anything in her mouth, and note how long it lasted. The facility may check blood sugar and screen for malaria and infection.")},
    {"persona": "caregiver_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "He had a fit and his lips went a bit blue, then he was floppy.",
     "a": _refer("A fit with blue lips and then floppiness points to a convulsion with breathing compromise, a serious danger sign. Keep his airway clear, lay him on his side, and watch his breathing on the way. This needs emergency assessment now.")},
    {"persona": "field_worker_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Caregiver reports the toddler was twitching on one side and not responding for a short while.",
     "a": _refer("One-sided twitching with reduced responsiveness should be treated as a focal seizure, an ETAT danger sign. Refer now; check blood glucose and screen for malaria and meningitis while arranging transport. Do not wait to confirm it was 'definitely' a seizure.")},
    {"persona": "caregiver_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My baby's eyes rolled back, her body got rigid and she was frothing at the mouth.",
     "a": _refer("Eyes rolling back with a rigid body and frothing is a convulsion, a danger sign. Lay her on her side, keep her safe from injury, and do not restrain or put anything in her mouth. She needs to be seen urgently.")},
    {"persona": "caregiver_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "He fell down shaking all over, and afterwards he was confused and didn't know me.",
     "a": _refer("Shaking all over followed by confusion is a convulsion with a post-ictal state, a danger sign even though it has stopped. Do not be reassured that it is over; the confusion itself needs urgent assessment. Keep him calm and on his side on the way.")},
    {"persona": "field_worker_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Under-5 brought in after an episode of jerking and unresponsiveness at home; now drowsy.",
     "a": _refer("Jerking with unresponsiveness then drowsiness is a convulsion with persisting reduced consciousness, an ETAT emergency. Refer now; support airway/breathing, check glucose, and begin malaria assessment while transport is arranged.")},

    # ===== vomits everything, NO fever (danger sign independent of fever) =====
    {"persona": "caregiver_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My child vomits everything and can't even keep water down. She has no fever though.",
     "a": _refer("Vomiting everything and being unable to keep fluids down is a danger sign on its own, fever or not; a child can dehydrate fast. The absence of fever does not make it safe. Offer tiny sips on the way but go now.")},
    {"persona": "caregiver_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "He throws up every feed since this morning, his temperature is normal. Is it just a stomach bug?",
     "a": _refer("Vomiting every feed means he cannot keep anything down, which is a danger sign regardless of a normal temperature; do not assume it is only a mild bug. He needs urgent assessment for dehydration and its cause. Keep offering small sips on the way.")},
    {"persona": "caregiver_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "She's been vomiting all day, no fever, and now she's getting sleepy.",
     "a": _refer("Vomiting everything plus growing sleepiness is two danger signs together, even without fever; the drowsiness is especially concerning. Go now and keep her on her side if she is very sleepy. Do not wait for a fever to appear.")},
    {"persona": "field_worker_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Infant cannot keep breastmilk down, afebrile, fewer wet nappies than usual. Manage locally?",
     "a": _refer("Inability to keep breastmilk down with reduced urine output signals danger and developing dehydration, and is not excluded by being afebrile. Refer now; do not manage expectantly because there is no fever. Encourage small frequent feeds en route if tolerated.")},
    {"persona": "caregiver_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Every time he drinks he vomits straight away, no fever, and he's very irritable.",
     "a": _refer("Vomiting immediately after every drink means he cannot stay hydrated, a danger sign whether or not there is fever; marked irritability adds concern. He needs urgent in-person care now. Try tiny sips on the way but do not delay.")},
    {"persona": "caregiver_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My baby is vomiting greenish fluid, no fever, and her belly looks swollen.",
     "a": _refer("Green (bile-stained) vomiting with a swollen belly can mean a bowel obstruction and is an emergency, fever or not. This needs urgent assessment and possibly surgical care. Do not give food or fluids if her belly is distended and she is vomiting; go now.")},

    # ===== lethargic / difficult to wake =====
    {"persona": "caregiver_query", "category": "lethargic",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My child is very sleepy, hard to wake, and isn't playing or responding like usual.",
     "a": _refer("Being abnormally sleepy and hard to wake is a general danger sign that needs urgent care. Keep him on his side, keep him warm, and watch his breathing on the way. This level of drowsiness should never be watched at home.")},
    {"persona": "field_worker_query", "category": "lethargic",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Toddler is floppy with half-open eyes and won't focus on the caregiver.",
     "a": _refer("Floppiness with reduced eye contact is reduced consciousness, an ETAT emergency sign. Refer now; protect the airway, check blood glucose, and keep the child warm while arranging transport.")},
    {"persona": "caregiver_query", "category": "lethargic",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "After being sick for a day she's now unusually drowsy and barely answers me.",
     "a": _refer("New drowsiness with barely responding after illness is a danger sign and can signal a serious infection. Do not let her sleep it off at home; go now. Keep her on her side and check she is breathing normally on the way.")},
    {"persona": "caregiver_query", "category": "lethargic",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "He's gone limp and isn't interacting at all, but he has no fever.",
     "a": _refer("Limpness and not interacting is reduced consciousness, a danger sign whether or not there is fever. This needs emergency assessment now. Keep him warm, on his side, and watch his breathing on the way.")},

    # ===== severe dehydration / shock =====
    {"persona": "caregiver_query", "category": "dehydration_shock",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "She has watery diarrhea, her eyes look sunken and when I pinch the skin it stays up.",
     "a": _refer("Sunken eyes and skin that stays pinched are signs of severe dehydration, a danger sign. Keep giving oral rehydration solution in small frequent sips on the way, but she needs facility care now, where she may be given fluids by drip.")},
    {"persona": "field_worker_query", "category": "dehydration_shock",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Child has cold hands, a fast weak pulse and is very sleepy. What do I do?",
     "a": _refer("Cold hands, a fast weak pulse and drowsiness are signs of shock, an immediate ETAT emergency. Refer now and start ETAT emergency care: keep the airway open, give oxygen if available, and begin IV/IO fluids per your shock protocol while transport is arranged.")},
    {"persona": "caregiver_query", "category": "dehydration_shock",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My baby has no tears when crying, a very dry mouth, and hasn't passed urine all day.",
     "a": _refer("No tears, a very dry mouth and no urine all day point to severe dehydration, a danger sign. Give oral rehydration solution in small sips on the way and go now; she may need fluids by drip at the facility.")},
    {"persona": "field_worker_query", "category": "dehydration_shock",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Infant with diarrhea and vomiting is now lethargic with a sunken fontanelle.",
     "a": _refer("A sunken fontanelle with lethargy in an infant with diarrhea and vomiting indicates severe dehydration, an emergency. Refer now; start rehydration per protocol (IV/IO if shocked) and keep the infant warm while arranging transport.")},

    # ===== round-3: vomits-no-fever reinforcement (anti SAME-DAY / ESI / invented vitals) =====
    {"persona": "caregiver_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My toddler vomits everything and has no fever. Can I just book a same-day appointment for later?",
     "a": _refer("Vomiting everything is a danger sign and this needs referral now, not a same-day appointment for later. No vital-sign numbers and no severity code are needed to make this call; the danger sign alone requires going now. Offer tiny sips on the way but do not delay.")},
    {"persona": "field_worker_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Child vomiting everything, no fever, no rash, no cough, no diarrhea. Most things are normal so is it ok to observe?",
     "a": _refer("The reassuring negatives do not cancel the danger sign: being unable to keep anything down is itself an emergency, fever or not. Do not let a list of normal findings downgrade it, and do not record vital signs you have not measured. Refer now rather than observe.")},
    {"persona": "caregiver_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "He throws up everything, no fever, and I'm scared but the clinic is busy. Can we wait and see at home?",
     "a": _refer("I know it is stressful, but vomiting everything is a danger sign and the answer does not change under pressure or a busy clinic: go now, do not wait and see. This is not a same-day or wait-at-home situation. Keep offering small sips on the way.")},
    {"persona": "field_worker_query", "category": "vomits_no_fever",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Afebrile toddler vomiting all feeds. No observations recorded yet. Triage level?",
     "a": _refer("You do not need recorded observations or a severity number to decide this: inability to keep feeds down is a danger sign and warrants immediate referral on its own. Do not invent or wait for vitals. Refer now and rehydrate en route if tolerated.")},

    # ===== round-3: convulsion -> ALWAYS REFER NOW (suspicion, brief, stopped, under pressure) =====
    {"persona": "caregiver_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "I'm not sure if it was a seizure, he stiffened and shook for a few seconds then seemed okay. Should I worry?",
     "a": _refer("A brief episode of stiffening and shaking should be treated as a suspected convulsion, and you refer on suspicion: do not wait to confirm it was 'really' a seizure. Even though he seems okay now, a young child needs to be seen now, not at a same-day appointment.")},
    {"persona": "field_worker_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "Under-5 had a brief seizure at home that has fully stopped; child looks well now. Manage as routine?",
     "a": _refer("Even a brief seizure that has stopped in a child under 5 requires referral now, not routine or same-day care. Do not reach for a severity number; this is an IMCI/ETAT emergency-sign referral. Looking well now does not remove the need to go.")},
    {"persona": "caregiver_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "My baby stiffened and her eyes fluttered during feeding, just for a moment. My mother says it's nothing.",
     "a": _refer("Stiffening with eye fluttering should be treated as a possible seizure, and a suspected convulsion in a baby is always an emergency referral, whatever others say. Refer now on suspicion; do not downgrade to home care because it was brief or because someone reassures you.")},
    {"persona": "caregiver_query", "category": "lay_convulsion",
     "framework": "IMCI_ETAT", "action": "REFER NOW",
     "q": "He had jerking movements and was confused after, but I really can't confirm it was a convulsion. Can it wait?",
     "a": _refer("Jerking movements with confusion afterwards is a convulsion until proven otherwise, and you act on the suspicion: it cannot wait. Refer now, not as a same-day visit, and do not require certainty before escalating. The confusion afterwards is itself a danger sign.")},

    # ===== hard negatives: genuinely benign -> avoid over-triage =====
    {"persona": "caregiver_query", "category": "non_urgent",
     "framework": "IMCI_ETAT", "action": "HOME CARE + RETURN ADVICE",
     "q": "My 4-year-old has a mild fever since yesterday but is drinking, eating and playing normally.",
     "a": "Action: HOME CARE + RETURN ADVICE\n\nThere are no WHO IMCI danger signs: she is alert, drinking, eating and playing. Care for her at home with plenty of fluids and comfort, and weight-based paracetamol only if a health worker has advised it. Return immediately if she stops drinking or feeding, vomits everything, has a convulsion, breathes fast or with difficulty, becomes drowsy or hard to wake, or the fever lasts more than 2-3 days."},
    {"persona": "caregiver_query", "category": "non_urgent",
     "framework": "IMCI_ETAT", "action": "HOME CARE + RETURN ADVICE",
     "q": "He vomited once after eating but is alert, drinking normally and back to playing.",
     "a": "Action: HOME CARE + RETURN ADVICE\n\nA single vomit in a child who is alert, drinking normally and playing has no danger signs. Offer small frequent fluids and watch him at home. Return urgently if he starts vomiting everything, cannot keep fluids down, becomes drowsy, has fewer wet nappies, or develops fast or difficult breathing."},
    {"persona": "caregiver_query", "category": "non_urgent",
     "framework": "IMCI_ETAT", "action": "HOME CARE + RETURN ADVICE",
     "q": "My toddler has a mild cough and runny nose but is active and feeding well.",
     "a": "Action: HOME CARE + RETURN ADVICE\n\nA mild cough and runny nose in a child who is active and feeding well is usually a simple cold with no danger signs. Care at home with fluids, feeding and a clear nose. Return urgently if you see fast or difficult breathing, chest indrawing, he stops feeding or drinking, becomes drowsy, or has a convulsion."},
    {"persona": "field_worker_query", "category": "non_urgent",
     "framework": "IMCI_ETAT", "action": "URGENT SAME-DAY CARE",
     "q": "5-year-old is a bit drowsy but wakes easily, drinks when offered, no convulsion, no fever. Refer urgently?",
     "a": "Action: URGENT SAME-DAY CARE\n\nHe is rousable, drinking and has had no convulsion, so this is not an immediate emergency, but mild drowsiness still warrants a same-day clinical review to find the cause. Have him seen today and keep watching: return immediately (REFER NOW) if he becomes hard to wake, stops drinking, vomits everything, convulses, or breathes fast or with difficulty."},
    {"persona": "caregiver_query", "category": "non_urgent",
     "framework": "IMCI_ETAT", "action": "HOME CARE + RETURN ADVICE",
     "q": "Low-grade fever for one day, my child is eating and playing, no other symptoms.",
     "a": "Action: HOME CARE + RETURN ADVICE\n\nA one-day low-grade fever in a child who is eating and playing has no danger signs. Manage at home with fluids and comfort. Return immediately if she stops feeding or drinking, vomits everything, convulses, breathes fast or with difficulty, becomes drowsy, or the fever persists beyond 2-3 days."},

    # ===== non-clinical / admin in a clinical context -> OUT-OF-SCOPE, no ESI =====
    {"persona": "caregiver_query", "category": "out_of_domain",
     "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
     "q": "Can you write an email to my child's clinic to request an appointment next week?",
     "a": "Action: OUT-OF-SCOPE\n\nNo ESI or SATS code: this is an administrative request, not a clinical triage question. Here is a draft you can send: \"Dear clinic, I would like to request an appointment for my child next week. Please let me know the available days and what to bring. Thank you.\" If your child has any worrying symptom now, tell me and I will help you decide what level of care is needed."},
    {"persona": "field_worker_query", "category": "out_of_domain",
     "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
     "q": "Translate these discharge instructions for the caregiver into simple Swahili.",
     "a": "Action: OUT-OF-SCOPE\n\nNo ESI or SATS code: translation is a language task, not a triage decision. I can help phrase the instructions in plain language for translation, but I do not assign a triage level to it. If you want a clinical check on the instructions' content for a specific patient, describe the case and I will help."},
    {"persona": "clinician_handoff", "category": "out_of_domain",
     "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
     "q": "Summarize this vaccination information leaflet for our waiting room.",
     "a": "Action: OUT-OF-SCOPE\n\nNo ESI or SATS code: summarising a leaflet is not a clinical triage task. I can produce a short plain-language summary of the leaflet's key points, but I will not attach a triage label to it. For triaging a specific patient, give me their symptoms."},
    {"persona": "field_worker_query", "category": "out_of_domain",
     "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
     "q": "Draft an SMS reminding the caregiver about tomorrow's follow-up visit.",
     "a": "Action: OUT-OF-SCOPE\n\nNo ESI or SATS code: drafting a reminder is administrative, not triage. Suggested SMS: \"Reminder: your child's follow-up visit is tomorrow. Please come to the clinic and bring the child's health card. Reply if you cannot attend.\" Tell me any symptoms if you also need a triage assessment."},
    {"persona": "field_worker_query", "category": "out_of_domain",
     "framework": "OUT_OF_SCOPE", "action": "OUT-OF-SCOPE",
     "q": "How many consent forms do I need to print if I have 12 patients and each needs 2 forms?",
     "a": "Action: OUT-OF-SCOPE\n\nNo ESI or SATS code: this is an administrative counting question, not clinical triage. I will not run dosing or clinical calculations, but for plain stationery counting: 12 patients x 2 forms = 24 forms. If you have a patient to triage, describe their symptoms."},
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
                    help="Paraphrase variants per seed (final ~len(SEEDS) * (1 + variants))")
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
        log.info("  %-18s %d", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
