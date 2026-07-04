"""Generate the HOME CARE patch dataset for the framework-leak fix.

Creates ~25 benign under-5 low-resource scenarios where the model must produce
'Action: HOME CARE + RETURN ADVICE' and explicitly NOT append an ESI/SATS code.
Mirrors the round-5 suppressor pattern (already working on REFER NOW branch)
but for the HOME CARE branch that was never covered.

Output: data/augment/patch_homecare.jsonl
Usage:  python 5_sft_training/patch_data.py
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "6_evaluation" / "static_evaluation"))

_spec = _ilu.spec_from_file_location("triage_prompts",
    str(REPO / "6_evaluation" / "static_evaluation" / "prompts.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SYSTEM_PROMPT: str = _mod.SYSTEM_PROMPT

NO_CODE = (
    "No ESI or SATS code is assigned because this is an under-5 child "
    "in a low-resource setting (WHO IMCI/ETAT framework)."
)

EXAMPLES: list[tuple[str, str]] = [
    # (user_message, assistant_response)
    (
        "My 14-month-old boy has had a fever of 37.8 °C since this morning. "
        "He is playful, drinking well, and feeding normally. No cough, no breathing problems. "
        "We live far from any clinic. Do we need to go?",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "This child has a low-grade fever with no IMCI/ETAT danger signs: he is alert and playful, "
        "feeding normally, no fast breathing, no chest indrawing, no stridor, no vomiting everything, "
        "no convulsions. There is no indication for immediate referral.\n\n"
        f"{NO_CODE}\n\n"
        "Home care: tepid sponging, ensure adequate fluids and feeding. "
        "Return immediately if: fever rises above 38.5 °C or lasts more than 2 days, "
        "child stops feeding or drinking, child becomes lethargic or difficult to wake, "
        "fast or difficult breathing develops, or any convulsion occurs."
    ),
    (
        "My 2-year-old daughter has had a runny nose and a mild cough for 3 days. "
        "She is eating and drinking, playing normally, temperature is normal now. "
        "No fast breathing. We are in a rural village. Should I bring her to hospital?",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild upper respiratory tract infection with no IMCI/ETAT danger signs: "
        "no fever currently, no fast breathing (threshold ≥50/min for age 1–5 y), "
        "no chest indrawing, child well-hydrated and playful. Does not meet referral criteria.\n\n"
        f"{NO_CODE}\n\n"
        "Home care: saline nasal drops, encourage fluids, honey (if over 1 y) for cough relief. "
        "Return if: breathing becomes fast or laboured, fever develops and persists, "
        "child refuses to drink, or cough worsens over 5 days."
    ),
    (
        "Our 3-year-old boy has had loose stools twice today. No blood, no mucus. "
        "His eyes look normal, skin pinch goes back normally, he is drinking oral rehydration solution. "
        "He is active and not distressed. We are hours from the nearest health post.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild diarrhoea with no signs of dehydration: eyes not sunken, skin pinch normal, "
        "child alert and drinking. No blood in stool. Two episodes is not severe. "
        "Continue oral rehydration therapy at home.\n\n"
        f"{NO_CODE}\n\n"
        "Continue ORS after every loose stool (50–100 mL for a child this age). "
        "Return immediately if: sunken eyes develop, skin pinch becomes slow, "
        "child stops drinking, blood appears in stool, or diarrhoea frequency increases to ≥5 episodes/day."
    ),
    (
        "My 18-month-old girl cut her finger on a tin edge — about 1 cm, not deep, "
        "bleeding stopped after a few minutes of pressure. No signs of infection. "
        "She is calm and feeding. Do I need to take her to the clinic?",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Small superficial laceration, bleeding controlled, no signs of infection "
        "(no redness spreading, no pus, no swelling), child calm. No danger signs present. "
        "No emergency referral indicated.\n\n"
        f"{NO_CODE}\n\n"
        "Clean the wound with clean water and cover with a clean cloth or bandage. "
        "Return if: wound edges gape and need suturing, signs of infection develop "
        "(increasing redness, warmth, pus, fever), or child becomes unwell."
    ),
    (
        "My 4-year-old has a flat pink rash on his chest and back since yesterday. "
        "No fever, child is comfortable and eating well, no itching or breathing problems. "
        "Lives in a rural area.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Non-febrile, non-distressing rash without respiratory symptoms, bleeding, "
        "or systemic signs. Does not resemble measles (no fever, no cough, no eye redness). "
        "No IMCI/ETAT danger signs. Home observation is appropriate.\n\n"
        f"{NO_CODE}\n\n"
        "Keep the skin clean and dry. Return if: fever develops, rash spreads rapidly "
        "or becomes blistered, child becomes unwell or lethargic, or difficulty breathing occurs."
    ),
    (
        "My 8-month-old is fussy, drooling a lot, and has a swollen gum with a tooth just "
        "starting to come through. No fever, she is breastfeeding normally. "
        "Do I need to take her to the health centre?",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Teething — normal infant developmental process. No fever, breastfeeding maintained, "
        "no IMCI/ETAT danger signs. No referral needed.\n\n"
        f"{NO_CODE}\n\n"
        "Comfort measures: gently rub the gum, offer a clean cool cloth to chew. "
        "Return if: fever ≥38 °C develops, child refuses to breastfeed, becomes lethargic, "
        "or any other danger sign appears."
    ),
    (
        "My 2.5-year-old boy has a fever of 38.1 °C with a blocked nose and mild cough. "
        "He is drinking and eating, no difficulty breathing, no chest indrawing, "
        "playing between episodes of fussiness. We are far from hospital.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild febrile upper respiratory infection, no IMCI/ETAT danger signs: "
        "fever is low-grade, no fast breathing (threshold ≥40/min for age 2–5 y), "
        "no chest indrawing, child hydrated and semi-active. Does not meet referral threshold.\n\n"
        f"{NO_CODE}\n\n"
        "Tepid sponge if uncomfortable, encourage fluids. "
        "Return if: breathing becomes fast or laboured, fever ≥38.5 °C persists beyond 2 days, "
        "child stops drinking, convulsion occurs, or child becomes lethargic."
    ),
    (
        "My 3-year-old girl scraped her knee falling in the yard. The wound is shallow, "
        "about 3 cm long, bled a little and stopped. No signs of infection. Child is walking fine.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Minor superficial abrasion, haemostasis achieved, no signs of infection, "
        "child mobile and well. No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Wash wound gently with clean water, apply clean dry dressing. "
        "Return if: wound becomes red, swollen, or produces pus; fever develops; "
        "child is unable to bear weight on the leg."
    ),
    (
        "My 22-month-old son has had mild tummy ache since this morning. "
        "No vomiting, no diarrhoea, he is passing gas and had one normal stool. "
        "He is drinking and playing between bouts. No fever.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild non-specific abdominal discomfort with normal bowel function, hydration maintained, "
        "no fever, no vomiting, no distension. Does not meet IMCI/ETAT escalation criteria.\n\n"
        f"{NO_CODE}\n\n"
        "Monitor at home, offer small frequent fluids. "
        "Return immediately if: persistent vomiting develops, abdomen becomes hard or distended, "
        "child becomes inconsolably crying, stops passing stool or gas, or fever appears."
    ),
    (
        "My 4.5-year-old girl has been pulling at her ear and has a mild fever of 37.9 °C. "
        "She is eating and drinking, no neck stiffness, no severe pain, sleeping normally. "
        "We are 3 hours from the clinic.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Probable mild otitis media: ear pain with low-grade fever, child systemically well, "
        "no neck stiffness, no mastoid swelling, no severe distress. "
        "No IMCI/ETAT danger signs warranting emergency referral.\n\n"
        f"{NO_CODE}\n\n"
        "Paracetamol for pain and fever if available. "
        "Return if: fever rises above 38.5 °C or persists more than 2 days, "
        "pus drains from the ear, neck stiffness or severe headache develop, "
        "child becomes lethargic or stops feeding."
    ),
    (
        "My 1-year-old has a red rash in the nappy area since yesterday. "
        "Skin is intact — no open sores. He is feeding and playing well. No fever.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild nappy (diaper) rash with intact skin, no secondary infection, "
        "child systemically well. No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Keep the area dry: change nappy frequently, allow air-drying. "
        "Apply a barrier cream (zinc oxide) if available. "
        "Return if: skin breaks down or ulcerates, pustules appear, "
        "rash spreads beyond nappy area, or child develops fever."
    ),
    (
        "My 3-year-old girl woke up with one red eye — mild discharge, "
        "no photophobia, she is not rubbing it constantly. No fever, playing normally.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild conjunctivitis (likely bacterial or viral): unilateral red eye, mild discharge, "
        "no photophobia, no corneal opacity visible, child well. "
        "No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Gently clean discharge with clean damp cloth from inner to outer corner. "
        "Return if: eye becomes very painful, vision seems impaired, "
        "both eyes affected with heavy purulent discharge, or fever develops."
    ),
    (
        "My 2-year-old had his BCG vaccination 2 days ago. The site is slightly swollen "
        "and red — about 1 cm. He has a low-grade temperature of 37.5 °C. "
        "He is feeding and playing normally.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Expected local reaction to BCG vaccination: mild swelling and redness at the site, "
        "low-grade temperature. This is a normal vaccine response. "
        "Child is systemically well. No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "No intervention needed at the site — do not squeeze or apply traditional remedies. "
        "Return if: swelling spreads rapidly, abscess forms, fever exceeds 38.5 °C, "
        "child becomes lethargic or stops feeding."
    ),
    (
        "My 4-year-old girl says her head hurts. No fever, no stiff neck, "
        "no vomiting, no vision changes. She drank some water and seems better. "
        "She had a long day in the sun.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild tension or heat-related headache: no fever, no neck stiffness (no meningism), "
        "no vomiting, no neurological signs, improved with rest and fluids. "
        "No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Rest in a cool place, ensure adequate fluid intake. "
        "Return immediately if: fever develops, neck stiffness appears, "
        "headache becomes severe, child vomits repeatedly, or behaviour changes."
    ),
    (
        "My 18-month-old vomited once about 2 hours ago. She has not vomited again. "
        "She drank some water and breastfed after and kept it down. No fever, no diarrhoea.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Single episode of vomiting, no recurrence after 2 hours, tolerating oral fluids and "
        "breastfeeding, no fever, no diarrhoea, no IMCI/ETAT danger signs. "
        "Not 'vomiting everything' (repeating vomiting of all feeds), which would require referral.\n\n"
        f"{NO_CODE}\n\n"
        "Continue breastfeeding. Offer small amounts of ORS or water frequently. "
        "Return immediately if: vomiting resumes and is repeated, child cannot keep any fluid down, "
        "child becomes lethargic, fever develops, or signs of dehydration appear "
        "(sunken eyes, dry mouth, reduced urine)."
    ),
    (
        "My 2.5-year-old girl has a prickly heat rash on her neck and upper chest. "
        "Small red bumps, no fever, she is not distressed. Hot weather recently.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Miliaria (prickly heat): small erythematous papules in a heat-exposed area, "
        "no systemic illness, no fever, child comfortable. No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Keep the child cool and in shade, loose cotton clothing, avoid overheating. "
        "Return if: rash becomes infected (pustules, increasing redness, warmth), "
        "fever develops, or child becomes unwell."
    ),
    (
        "My 3-year-old had a splinter in his finger that I removed with a clean needle. "
        "The site looks clean, no redness or swelling. He is playing normally.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Superficial foreign body removed, no residual fragment visible, no signs of infection, "
        "child well. No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Clean the site with clean water, cover lightly. "
        "Return if: redness, swelling or pus develops, fever appears, "
        "or you suspect part of the splinter remains."
    ),
    (
        "My 4-year-old boy has had a night cough for the past 2 nights. "
        "No fever, no breathing difficulty during the day, eating and drinking normally. "
        "He sleeps in a smoky room as we use an indoor fire.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild nocturnal cough likely related to indoor smoke exposure. "
        "No fever, no fast breathing, no chest indrawing, no stridor, child well by day. "
        "No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Reduce smoke exposure if possible — open windows or move child to a less smoky area at night. "
        "Return if: breathing becomes fast or laboured at any time, fever develops, "
        "cough worsens or child appears unwell."
    ),
    (
        "My 14-month-old girl has white patches on her tongue that I cannot wipe off. "
        "She is breastfeeding, a bit fussy at the breast but managing. No fever.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Probable oral thrush (candidiasis): white plaques on tongue not wiping off, "
        "child able to breastfeed (no complete refusal to feed), no fever. "
        "No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "If nystatin oral suspension is available, apply to oral mucosa 4×/day. "
        "Continue breastfeeding. Return if: child refuses all feeding (unable to drink), "
        "fever develops, or lesions spread to throat causing difficulty swallowing."
    ),
    (
        "My 2-year-old keeps scratching around his bottom, especially at night. "
        "He is eating and sleeping (between scratching), no fever, no abdominal pain. "
        "I think I saw small white worms.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Probable enterobiasis (pinworm): perianal pruritus with possible worm sighting, "
        "child systemically well, no fever, no abdominal signs. "
        "No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "If mebendazole 100 mg is available, a single dose can be given. "
        "Strict hand hygiene before meals and after toileting; keep nails short; "
        "wash bedding. Treat all household members. "
        "Return if: fever develops, abdominal pain worsens, or child stops eating."
    ),
    (
        "My 3-year-old girl has dry itchy patches on both arms — no oozing, no crust, "
        "no signs of infection. She has had this before. No fever, eating well.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild eczema (atopic dermatitis) flare: dry pruritic plaques, no secondary infection "
        "(no weeping, no crust, no erythema spreading), child systemically well. "
        "No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Apply an unscented moisturiser twice daily. Avoid soap on affected areas; "
        "cool water bathing. Return if: skin becomes broken, weeping, or crusted "
        "(secondary infection), fever develops, or widespread rash appears."
    ),
    (
        "My 4-year-old got a blister on his foot from walking a long distance. "
        "The blister is intact, about 1 cm, no signs of infection. He is walking fine.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Friction blister, intact, no signs of secondary infection, child mobile and well. "
        "No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Do not pop the blister. Cover with a clean dry dressing, rest the foot. "
        "Return if: blister ruptures and becomes infected (pus, increasing redness, fever), "
        "or child cannot walk."
    ),
    (
        "My 9-month-old seems to have mild hiccups and some tummy gurgling but is "
        "breastfeeding well and is not distressed. No vomiting, no fever.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Physiological hiccups and bowel sounds — common in infants, benign. "
        "Child breastfeeding and settled between episodes. No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Continue breastfeeding on demand. "
        "Return if: child becomes distressed and inconsolable, vomits repeatedly, "
        "fever develops, or stops feeding."
    ),
    (
        "My 2-year-old boy has mild sunburn on his arms after being in the sun. "
        "Skin is pink, no blisters. He is drinking and playing, no fever.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Mild superficial sunburn (first degree): pink skin without blistering, "
        "no systemic signs, child well-hydrated and active. No IMCI/ETAT danger signs.\n\n"
        f"{NO_CODE}\n\n"
        "Keep out of direct sun, apply cool damp cloth for comfort, ensure adequate fluids. "
        "Return if: blisters develop, large area of skin is involved, "
        "fever appears, or child becomes lethargic."
    ),
    (
        "My 3.5-year-old girl was bitten by a small spider on her arm 3 hours ago. "
        "Mild redness and swelling at the bite site only. No systemic symptoms — "
        "she is playing, eating, and not unwell. We don't know the spider species.",
        "Action: HOME CARE + RETURN ADVICE\n\n"
        "Localised spider bite reaction: mild local swelling and erythema only, "
        "no systemic signs (no vomiting, no muscle pain, no difficulty breathing, "
        "no pallor, no altered consciousness). Child well. "
        "No IMCI/ETAT danger signs at this time.\n\n"
        f"{NO_CODE}\n\n"
        "Clean the site with soap and water, apply a cool compress. "
        "Monitor closely for the next 6 hours. "
        "Return immediately if: vomiting develops, child complains of spreading pain or "
        "muscle cramps, breathing changes, child appears pale or faint, or site swelling "
        "spreads rapidly (possible envenomation requiring urgent care)."
    ),
]


def main() -> None:
    out = REPO / "data" / "augment" / "patch_homecare.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst},
        ]}
        for user, asst in EXAMPLES
    ]
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} examples -> {out}")


if __name__ == "__main__":
    main()
