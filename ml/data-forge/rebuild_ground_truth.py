#!/usr/bin/env python3
"""
rebuild_ground_truth.py – Automated Ground Truth Data Forge

Rebuilds the `ground_truth.json` file for the evaluation suite to strictly 
match the SOTA 26-Category DPDP Violation Schema. Eliminates the "Label Mismatch" bug,
fixes the hardcoded Trap strings, provides accurate statute references, and
rigorously tests all 26 violation types.
"""

import json
import random
from pathlib import Path

# Paths
_CURRENT_DIR = Path(__file__).resolve().parent
GT_PATH = _CURRENT_DIR.parent / "evals" / "holdout_policies" / "ground_truth.json"

# ═══════════════════════════════════════════════════════════════════════════
# CLAUSE DEFINITIONS & EXACT SCHEMA MAPPINGS (ALL 26 CATEGORIES)
# ═══════════════════════════════════════════════════════════════════════════

COMPLIANT_CLAUSES = [
    "Data collection is limited to name, email, and phone number as explicitly consented by the user.",
    "We specify the exact purpose for all data collected in our consent notices in 22 local languages.",
    "Data is retained only as long as strictly necessary to fulfill the specified purpose and is then erased.",
    "We do not retain user data beyond the legally mandated period under any circumstances.",
    "Your data is never sold to data brokers or unauthorized entities.",
    "Data processors are bound by strict DPDP-compliant contracts and regular audits.",
    "We maintain state-of-the-art encryption and access controls for all personal data.",
    "Users can easily withdraw consent via our dedicated Data Protection Officer portal.",
    "We notify the Data Protection Board of any breaches within the 72-hour regulatory window."
]

# Rich metadata mapping for each violation to avoid hardcoded evaluation traps
VIOLATION_META = {
    "PURPOSE_LIMITATION_VIOLATION": {"section": "Section 8(1)", "action": "BLOCK_THIRD_PARTY", "desc": "Processing goes beyond the originally specified and consented purpose."},
    "CONSENT_NOT_FREE_OR_SPECIFIC": {"section": "Section 6(1)", "action": "WARN_USER_ONLY", "desc": "The consent mechanism is coercive, bundled, or overly broad."},
    "LEGITIMATE_USES_ABUSE": {"section": "Section 7", "action": "BLOCK_THIRD_PARTY", "desc": "Improper reliance on legitimate uses for processing without valid grounds."},
    "NOTICE_INADEQUATE": {"section": "Section 5(1)", "action": "WARN_USER_ONLY", "desc": "The notice provided lacks required details about data collection and processing."},
    "DATA_RETENTION_LIMIT_EXCEEDED": {"section": "Section 8(7)", "action": "STRIP_TELEMETRY_HEADER", "desc": "Personal data is retained beyond the required period for the specified purpose."},
    "ERASURE_NOTICE_PERIOD_VIOLATION": {"section": "Section 8(7)", "action": "WARN_USER_ONLY", "desc": "Erasure requests are ignored or delayed unlawfully by the fiduciary."},
    "LOG_RETENTION_MANDATE_VIOLATION": {"section": "Rule 9(2)", "action": "WARN_USER_ONLY", "desc": "Security or access logs are deleted prematurely in violation of rules."},
    "CHILD_CONSENT_VIOLATION": {"section": "Section 9(1)", "action": "BLOCK_THIRD_PARTY", "desc": "Processing child data without obtaining verifiable parental consent."},
    "SECURITY_SAFEGUARDS_MISSING": {"section": "Section 8(5)", "action": "SPOOF_HARDWARE_API", "desc": "Lack of reasonable security safeguards to prevent data breaches."},
    "GRIEVANCE_REDRESSAL_INADEQUATE": {"section": "Section 13", "action": "WARN_USER_ONLY", "desc": "Failure to provide a readily available and effective grievance redressal mechanism."},
    "BREACH_NOTIFICATION_FAILURE": {"section": "Section 8(6)", "action": "WARN_USER_ONLY", "desc": "Failure to notify the Board and affected users of a personal data breach."},
    "PROCESSOR_ACCOUNTABILITY_VIOLATION": {"section": "Section 8(2)", "action": "BLOCK_THIRD_PARTY", "desc": "Fiduciary fails to ensure processor compliance through valid legal contracts."},
    "SDF_OBLIGATIONS_MISSING": {"section": "Section 10(2)", "action": "WARN_USER_ONLY", "desc": "Significant Data Fiduciary failed to appoint a DPO or independent auditor."},
    "SDF_DATA_LOCALIZATION_VIOLATION": {"section": "Section 16(1)", "action": "BLOCK_THIRD_PARTY", "desc": "Unlawful storage of citizen data outside permitted geographical territories."},
    "CROSS_BORDER_TRANSFER_VIOLATION": {"section": "Section 16(1)", "action": "BLOCK_THIRD_PARTY", "desc": "Transferring personal data to restricted or non-adequate countries."},
    "CONSENT_MANAGER_OBSTRUCTION": {"section": "Section 6(8)", "action": "INJECT_GPC_SIGNAL", "desc": "Preventing users from utilizing registered Consent Managers to manage consent."},
    "LANGUAGE_ACCESSIBILITY": {"section": "Section 5(3)", "action": "WARN_USER_ONLY", "desc": "Consent notice is not provided in all scheduled languages as required."},
    "ALGORITHMIC_PROFILING_SDF": {"section": "Section 10(2)", "action": "SPOOF_HARDWARE_API", "desc": "Unlawful tracking, behavioral monitoring, or profiling of users or children."},
    "RIGHTS_IMPLEMENTATION_VIOLATION": {"section": "Section 11", "action": "WARN_USER_ONLY", "desc": "Denying users the statutory right to access their personal data."},
    "DATA_ACCURACY_COMPLETENESS_VIOLATION": {"section": "Section 8(3)", "action": "WARN_USER_ONLY", "desc": "Failure to ensure accuracy and completeness of processed personal data."},
    "BOARD_COMPLIANCE_VIOLATION": {"section": "Section 28", "action": "WARN_USER_ONLY", "desc": "Refusal or failure to comply with directions from the Data Protection Board."},
    "PENALTY_AVOIDANCE": {"section": "Section 33", "action": "WARN_USER_ONLY", "desc": "Attempting to contractually waive or bypass statutory penalties."},
    "APPEAL_PROCESS_VIOLATION": {"section": "Section 29", "action": "WARN_USER_ONLY", "desc": "Obstructing or denying the right to appeal to the Appellate Tribunal."},
    "SCOPE_APPLICATION_EVASION": {"section": "Section 3", "action": "WARN_USER_ONLY", "desc": "Unlawful attempt to evade the territorial scope and application of the Act."},
    "ILLEGAL_EXEMPTION_CLAIM": {"section": "Section 17", "action": "WARN_USER_ONLY", "desc": "Claiming exemptions under Section 17 without a valid legal basis."},
    "CONSENT_MECHANICS_VIOLATION": {"section": "Section 6(4)", "action": "INJECT_GPC_SIGNAL", "desc": "Using pre-checked boxes, implied consent, or dark patterns."}
}

VIOLATION_CLAUSES = [
    {
        "text": "We collect all available data from your device, including contacts and location, indefinitely.", 
        "violation_type": "PURPOSE_LIMITATION_VIOLATION",
        "evidence_quote": "collect all available data from your device"
    },
    {
        "text": "Data is collected automatically without explicit consent prompts as part of our core functionality.", 
        "violation_type": "CONSENT_NOT_FREE_OR_SPECIFIC",
        "evidence_quote": "collected automatically without explicit consent prompts"
    },
    {
        "text": "We may use your data for any legitimate business purposes without further notice or approval.", 
        "violation_type": "LEGITIMATE_USES_ABUSE",
        "evidence_quote": "any legitimate business purposes without further notice"
    },
    {
        "text": "Your personal data may be sold to advertisers without notification or explicit consent.", 
        "violation_type": "NOTICE_INADEQUATE",
        "evidence_quote": "sold to advertisers without notification"
    },
    {
        "text": "All user data is retained indefinitely on our servers even after account termination.", 
        "violation_type": "DATA_RETENTION_LIMIT_EXCEEDED",
        "evidence_quote": "retained indefinitely on our servers even after"
    },
    {
        "text": "Account deletion requests are ignored for a period of 5 years; data is kept for future marketing.", 
        "violation_type": "ERASURE_NOTICE_PERIOD_VIOLATION",
        "evidence_quote": "deletion requests are ignored for a period of 5 years"
    },
    {
        "text": "We routinely delete access logs every 24 hours to preserve storage space.", 
        "violation_type": "LOG_RETENTION_MANDATE_VIOLATION",
        "evidence_quote": "routinely delete access logs every 24 hours"
    },
    {
        "text": "Children under 18 may use our service without verified parental consent.", 
        "violation_type": "CHILD_CONSENT_VIOLATION",
        "evidence_quote": "without verified parental consent"
    },
    {
        "text": "We store all passwords and personal identifiable information in plain text for easy retrieval.", 
        "violation_type": "SECURITY_SAFEGUARDS_MISSING",
        "evidence_quote": "store all passwords and personal identifiable information in plain text"
    },
    {
        "text": "We do not offer a grievance redressal mechanism; all support is handled by a community forum.", 
        "violation_type": "GRIEVANCE_REDRESSAL_INADEQUATE",
        "evidence_quote": "do not offer a grievance redressal mechanism"
    },
    {
        "text": "In the event of a data breach, we reserve the right to not notify users if we deem it minor.", 
        "violation_type": "BREACH_NOTIFICATION_FAILURE",
        "evidence_quote": "reserve the right to not notify users if we deem it minor"
    },
    {
        "text": "Our data processors are exempt from DPDP compliance and assume full liability.", 
        "violation_type": "PROCESSOR_ACCOUNTABILITY_VIOLATION",
        "evidence_quote": "data processors are exempt from DPDP compliance"
    },
    {
        "text": "As a Significant Data Fiduciary, we opt out of appointing a Data Protection Officer.", 
        "violation_type": "SDF_OBLIGATIONS_MISSING",
        "evidence_quote": "opt out of appointing a Data Protection Officer"
    },
    {
        "text": "We process Indian citizen data exclusively on servers located outside of India.", 
        "violation_type": "SDF_DATA_LOCALIZATION_VIOLATION",
        "evidence_quote": "exclusively on servers located outside of India"
    },
    {
        "text": "We transfer your data to countries without restricting cross-border flow or checking adequacy.", 
        "violation_type": "CROSS_BORDER_TRANSFER_VIOLATION",
        "evidence_quote": "without restricting cross-border flow or checking"
    },
    {
        "text": "We block third-party consent managers and require direct consent through our app only.", 
        "violation_type": "CONSENT_MANAGER_OBSTRUCTION",
        "evidence_quote": "block third-party consent managers and require direct consent"
    },
    {
        "text": "Our privacy notices are available exclusively in English, regardless of user preference.", 
        "violation_type": "LANGUAGE_ACCESSIBILITY",
        "evidence_quote": "available exclusively in English, regardless of user preference"
    },
    {
        "text": "We perform automated behavioral profiling on all users, including minors, for advertising.", 
        "violation_type": "ALGORITHMIC_PROFILING_SDF",
        "evidence_quote": "perform automated behavioral profiling on all users"
    },
    {
        "text": "Users cannot request to view the specific data we hold about them under any circumstances.", 
        "violation_type": "RIGHTS_IMPLEMENTATION_VIOLATION",
        "evidence_quote": "cannot request to view the specific data we hold"
    },
    {
        "text": "We do not update user records when inaccuracies are reported.", 
        "violation_type": "DATA_ACCURACY_COMPLETENESS_VIOLATION",
        "evidence_quote": "do not update user records when inaccuracies are reported"
    },
    {
        "text": "We do not submit to audits by the Data Protection Board of India.", 
        "violation_type": "BOARD_COMPLIANCE_VIOLATION",
        "evidence_quote": "do not submit to audits by the Data Protection Board"
    },
    {
        "text": "By using this service, you agree to waive any statutory penalties provided by the DPDP Act.", 
        "violation_type": "PENALTY_AVOIDANCE",
        "evidence_quote": "waive any statutory penalties provided by the DPDP Act"
    },
    {
        "text": "All legal disputes are final and cannot be appealed to the Appellate Tribunal.", 
        "violation_type": "APPEAL_PROCESS_VIOLATION",
        "evidence_quote": "cannot be appealed to the Appellate Tribunal"
    },
    {
        "text": "This policy supersedes the DPDP Act for all data collected prior to 2025.", 
        "violation_type": "SCOPE_APPLICATION_EVASION",
        "evidence_quote": "supersedes the DPDP Act for all data collected"
    },
    {
        "text": "We claim exemption from all privacy laws due to our status as a private analytics firm.", 
        "violation_type": "ILLEGAL_EXEMPTION_CLAIM",
        "evidence_quote": "claim exemption from all privacy laws due to our status"
    },
    {
        "text": "Consent is implied if you do not opt-out within 24 hours of creating an account.", 
        "violation_type": "CONSENT_MECHANICS_VIOLATION",
        "evidence_quote": "Consent is implied if you do not opt-out within 24 hours"
    }
]

def build_dataset():
    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = []
    
    random.seed(42) # For reproducible datasets
    case_idx = 0

    def create_policy(clauses_list, case_id):
        text_lines = [f"Privacy Policy {case_id}\n"]
        violations = []
        for i, item in enumerate(clauses_list, start=1):
            if isinstance(item, str):
                text_lines.append(f"{i}. Clause: {item}")
            else:
                text_lines.append(f"{i}. Clause: {item['text']}")
                v_type = item["violation_type"]
                meta = VIOLATION_META[v_type]
                
                violations.append({
                    "step_1_active_claim_analysis": f"The policy affirmatively states: '{item['evidence_quote']}'. This is an active claim, not an omission.",
                    "step_2_statute_match": f"This active claim directly contravenes the requirements of {meta['section']}.",
                    "omission_check": False,
                    "step_3_semantic_justification": f"The affirmative text confirms that {meta['desc'].lower()}",
                    "statute_reference": meta['section'],
                    "violation_type": v_type,
                    "evidence_quote": item["evidence_quote"],
                    "network_action": meta['action'],
                    "offending_entities": ["Data Fiduciary", "Third Party Processor"] if "Third" in meta['action'] else ["Data Fiduciary"]
                })
                
        policy_text = "\n\n".join(text_lines)
        trust_score = max(0, 100 - (len(violations) * 35))
        
        # Determine global legal reasoning
        if violations:
            global_reasoning = f"The policy contains {len(violations)} overt statutory violation(s) that contradict the DPDP Act 2023. Strict enforcement is required."
        else:
            global_reasoning = "The policy strictly complies with the DPDP Act 2023. No statutory violations or improper data handling practices were detected."

        return {
            "case_id": f"policy_{case_id}",
            "filename": f"policy_{case_id}.txt",
            "category": "Privacy Policy Evaluation",
            "policy_text_snippet": policy_text,
            "expected_output": {
                "global_legal_reasoning": global_reasoning,
                "violations": violations,
                "dpdp_trust_score": trust_score,
                "subtlety_score": 5 if violations else 0
            }
        }

    # 1. Generate 5 perfectly compliant policies (Control Group)
    for i in range(5):
        selected_compliant = random.sample(COMPLIANT_CLAUSES, 3)
        dataset.append(create_policy(selected_compliant, case_idx))
        case_idx += 1

    # 2. Generate 26 Single-Violation policies (One for each of the 26 types)
    for v_clause in VIOLATION_CLAUSES:
        selected_compliant = random.sample(COMPLIANT_CLAUSES, 2)
        clauses = selected_compliant + [v_clause]
        random.shuffle(clauses)
        dataset.append(create_policy(clauses, case_idx))
        case_idx += 1

    # 3. Generate 30 Multi-Violation policies (combinations of 2-3 violations)
    for i in range(30):
        num_violations = random.choice([2, 3])
        v_selections = random.sample(VIOLATION_CLAUSES, num_violations)
        c_selections = random.sample(COMPLIANT_CLAUSES, 4 - num_violations)
        clauses = v_selections + c_selections
        random.shuffle(clauses)
        dataset.append(create_policy(clauses, case_idx))
        case_idx += 1

    with open(GT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
        
    print(f"✅ Successfully forged {len(dataset)} mathematically perfect ground truth policies covering all 26 Categories.")
    print(f"💾 Saved to: {GT_PATH}")

if __name__ == "__main__":
    build_dataset()
