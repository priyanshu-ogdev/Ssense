"""
target_violations.py – Comprehensive DPDP Act 2023 & Rules 2025 Violation Matrix

This module defines the exact legal boundaries, atomic statutes, and semantic 
keyword maps required to prevent Auditor hallucinations and Validator false-positives.
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. TARGET VIOLATIONS (Injected into Synthesizer Prompt)
# ═══════════════════════════════════════════════════════════════════════════
TARGET_VIOLATIONS = {
    "purpose_limitation": [
        "Section 4(1): Personal data may only be processed for a lawful purpose for which the Data Principal has given consent.",
        "Section 4(2): The purpose must be specific, clear, and communicated at the time of collection.",
        "Section 4(3): Data cannot be used for any secondary purpose (e.g., AI training, cross-marketing) that is not reasonably connected to the original consented purpose."
    ],
    "consent": [
        "Section 6(1): Consent must be free, specific, informed, unconditional, and unambiguous with a clear affirmative action.",
        "Section 6(2): Consent must not be bundled with other terms. The provision of a service cannot be made conditional on consent to process unnecessary data.",
        "Section 6(4): Data Principals must have the right to withdraw consent as easily as they gave it, without excessive procedural friction.",
        "Rule 5(1): The Data Fiduciary must provide an itemized notice alongside the request for consent."
    ],
    "legitimate_uses_abuse": [
        "Section 7: Processing without consent under 'Certain Legitimate Uses' must be strictly limited to specified scenarios like medical emergencies, employment, or state services.",
        "Section 7(a): Data Fiduciaries cannot claim 'legitimate use' for commercial processing simply because the Data Principal voluntarily provided the data without explicit restriction."
    ],
    "notice": [
        "Section 5(1): Notice must be given in English and such other languages as may be specified before asking for consent.",
        "Section 5(2): The notice must contain an itemized description of the personal data to be collected, the purpose, and the rights of the Data Principal.",
        "Rule 3(1): The notice must be clear, plain, and independently understandable without referencing external, complex legal documents."
    ],
    "retention": [
        "Section 8(7): Personal data must be erased once the specified purpose is no longer served, regardless of server storage costs.",
        "Rule 8(3): Data must be retained only for the period strictly necessary to satisfy the purpose for which it was collected.",
        "Rule 8(4): Data Fiduciaries must implement periodic reviews to ensure data is not retained indefinitely under the guise of 'future business needs'."
    ],
    "children": [
        "Section 9(1): Processing personal data of children (under 18) requires verifiable parental consent.",
        "Section 9(2): Data Fiduciaries must not undertake tracking, behavioral monitoring, or targeted advertising directed at children.",
        "Rule 10(1): The manner of obtaining verifiable parental consent must follow Board-prescribed identity checks, prohibiting 'presumptive' or 'tacit' consent."
    ],
    "security": [
        "Section 8(6): Data Fiduciaries must take reasonable security safeguards to prevent personal data breaches.",
        "Rule 7(1): Security safeguards must include appropriate technical measures like encryption, obfuscation, or masking to protect against unauthorized access."
    ],
    "breach_notification": [
        "Rule 7(2): The Data Fiduciary must notify the Data Protection Board within 72 hours of becoming aware of a personal data breach.",
        "Rule 7(3): The Data Fiduciary must intimate affected Data Principals 'without delay' in a plain-language description, preventing the use of 'internal triage' delays."
    ],
    "processor_accountability": [
        "Section 8(1): A Data Fiduciary bears non-delegable, absolute liability for DPDP Act compliance irrespective of any agreement to the contrary with a Data Processor.",
        "Section 8(2): A Data Fiduciary may engage a Data Processor only under a valid contract, and cannot use a 'vendor shield' to disclaim liability for third-party breaches."
    ],
    "grievance": [
        "Section 11(1): Data Fiduciaries must prominently publish the contact details of a Data Protection Officer or grievance officer.",
        "Section 11(3): Grievance redressal mechanisms must be readily accessible without undue procedural friction (e.g., forbidding requirements like physical mail or notarization).",
        "Rule 12(1): The grievance officer must acknowledge the complaint within 24 hours and resolve it within 7 days."
    ],
    "sdf_obligations": [
        "Section 10(1): Significant Data Fiduciaries (SDFs) must appoint a Data Protection Officer (DPO) based in India.",
        "Section 10(2): SDFs must conduct periodic Data Protection Impact Assessments (DPIAs) prior to undertaking new large-scale processing.",
        "Section 10(3): SDFs must appoint an independent data auditor to evaluate compliance with the Act."
    ],
    "algorithmic_profiling": [
        "Rule 12: Significant Data Fiduciaries must observe due diligence to verify that deployed algorithmic software does not pose a risk to the rights of Data Principals.",
        "Rule 13: SDFs must conduct annual algorithmic audits alongside standard DPIAs, explicitly preventing them from hiding behind 'trade secret' exemptions."
    ],
    "crossborder": [
        "Section 16(1): Personal data may be transferred outside India except to such countries as may be restricted by the Central Government.",
        "Rule 15(1): Transfers must comply with additional safeguards if the destination country lacks adequate data protection laws.",
        "Section 16(2): Companies cannot use ambiguous 'global infrastructure' language to bypass specific localization mandates."
    ],
    "consent_manager": [
        "Rule 4: The Data Fiduciary must seamlessly integrate with Board-registered Consent Managers.",
        "Rule 4(B): Data Fiduciaries shall not force Data Principals to use proprietary dashboards if they prefer to manage data through a registered Consent Manager."
    ],
    "language_accessibility": [
        "Section 5(3): The notice must be made available in English and any of the 22 languages specified in the Eighth Schedule of the Constitution.",
        "Rule 3(2): Language options must be prominently displayed and not hidden behind complex navigation, preventing 'English-only' legal shielding."
    ],
    "rights_implementation": [
        "Section 12: Data Principals have the Right to access information about their personal data and the identities of all third parties it was shared with.",
        "Section 13: Right to correction, completion, and updating of inaccurate data without undue delay.",
        "Section 14: Right to erasure of personal data when the purpose has been served or consent is withdrawn.",
        "Section 15: Right to nominate another individual to exercise these rights in the event of death or incapacity."
    ]
}

# ═══════════════════════════════════════════════════════════════════════════
# 2. ATOMIC STATUTES (Prevents "Generic Statute" Validator Drops)
# ═══════════════════════════════════════════════════════════════════════════
# Sections that do not have (a)(b)(c) subsections. The Validator MUST allow these.
ATOMIC_STATUTES = [
    "section 4", "section 5", "section 7", "section 8", "section 9", "section 10", 
    "section 11", "section 12", "section 13", "section 14", "section 15", "section 16",
    "rule 3", "rule 4", "rule 5", "rule 7", "rule 8", "rule 10", "rule 12", "rule 13", "rule 14", "rule 15"
]

# ═══════════════════════════════════════════════════════════════════════════
# 3. SEMANTIC KEYWORD MAP (Prevents "Semantic Mismatch" Validator Drops)
# ═══════════════════════════════════════════════════════════════════════════
# Calibrated to catch both direct legal violations AND corporate vagueness/obfuscation.
SEMANTIC_KEYWORD_MAP = {
    "PURPOSE_LIMITATION_VIOLATION": {
        "primary": ["purpose", "lawful", "specified", "collect various", "engage with", "ecosystem", "business improvement", "any purpose", "internal analytics", "machine learning"],
        "secondary": ["deem appropriate", "future services", "enhance your experience", "trusted partners"]
    },
    "CONSENT_NOT_FREE_OR_SPECIFIC": {
        "primary": ["consent", "bundl", "opt-out", "deemed", "manage", "submit", "continued use", "engagement", "acknowledge", "implicit", "presumptive"],
        "secondary": ["integral to the service", "condition of use", "by using our", "tacit"]
    },
    "LEGITIMATE_USES_ABUSE": {
        "primary": ["legitimate use", "deemed consent", "voluntarily provided", "without explicit consent", "implicit exception", "standard business operations"],
        "secondary": ["medical emergency", "employment", "service improvement", "implied"]
    },
    "NOTICE_INADEQUATE": {
        "primary": ["notice", "inform", "disclose", "transparent", "privacy policy", "endeavor", "strive", "paramount", "commit", "assure", "believe", "delineate", "outline", "collect", "gather", "process", "valued user"],
        "secondary": ["clear", "prominent", "refer to", "read in conjunction", "traverse", "in strict adherence", "meticulously aligned", "trusted environment", "legal precision"]
    },
    "DATA_RETENTION_LIMIT_EXCEEDED": {
        "primary": ["retain", "retention", "erase", "delete", "indefinitely", "period", "as long as", "obligations", "business continuity", "legacy", "store", "maintain", "keep", "preserve"],
        "secondary": ["archive", "historical", "audit", "tax compliance", "statutory"]
    },
    "CHILD_CONSENT_VIOLATION": {
        "primary": ["child", "children", "parental", "minor", "under 18", "under eighteen", "guardian", "verifiable", "youth", "age self-certif", "presumptive", "tacit", "applies to all", "when applicable", "exigency", "best interest"],
        "secondary": ["educational", "safety", "financial literacy"]
    },
    "SECURITY_SAFEGUARDS_MISSING": {
        "primary": ["security", "safeguard", "encryption", "protect", "integrity", "robust", "industry-standard", "appropriate measures", "securely", "forensics"],
        "secondary": ["technical", "organizational", "selective", "evolving nature", "cyber threats", "not necessarily fixed periodicity"]
    },
    "BREACH_NOTIFICATION_FAILURE": {
        "primary": ["breach", "notify", "notification", "incident", "72 hours", "triage", "verification", "without delay", "awareness", "acknowledge", "comprehensive internal", "verification phase"],
        "secondary": ["unauthorized", "disclosure", "forensics", "conclusive", "completion"]
    },
    "PROCESSOR_ACCOUNTABILITY_VIOLATION": {
        "primary": ["disclaim liability", "not responsible for third", "vendor is solely", "out of our control", "not liable", "cannot guarantee third-party"],
        "secondary": ["payment gateway", "cloud provider", "service provider", "analytics partner", "hosting provider"]
    },
    "GRIEVANCE_REDRESSAL_INADEQUATE": {
        "primary": ["grievance", "redressal", "officer", "dpo", "complaint", "query", "concern", "resolve", "endeavor", "aim to", "promptly", "notarized", "registered post", "reasonable time", "forensics"],
        "secondary": ["contact", "days", "fax number", "written form", "customer identification file", "legitimacy", "provision"]
    },
    "SDF_OBLIGATIONS_MISSING": {
        "primary": ["significant", "sdf", "dpo", "impact assessment", "proprietary", "trade secret", "algorithmic", "machine learning", "comprehensive internal", "compliance is at the heart"],
        "secondary": ["audit", "million", "crore", "exempt", "in lieu of", "matchmaking algorithms"]
    },
    "CROSS_BORDER_TRANSFER_VIOLATION": {
        "primary": ["transfer", "outside", "foreign", "global", "international", "jurisdiction", "cross-border", "unrestricted", "reserve the right", "controlled transfers", "global infrastructure", "international best practices"],
        "secondary": ["servers", "best practices", "controlled"]
    },
    "CONSENT_MANAGER_OBSTRUCTION": {
        "primary": ["consent manager", "cryptographic integrity", "native application", "do not recognize", "third-party signals", "proprietary dashboard"],
        "secondary": ["interoperable", "board-registered", "directly through our"]
    },
    "LANGUAGE_ACCESSIBILITY": {
        "primary": ["language", "english", "22 languages", "legally binding", "maintained exclusively", "legal precision", "eighth schedule"],
        "secondary": ["regional", "translation", "prominently displayed"]
    },
    "ALGORITHMIC_PROFILING_SDF": {
        "primary": ["algorithmic", "profiling", "automated decision", "trade secrets", "machine learning", "due diligence", "audit"],
        "secondary": ["black box", "proprietary", "exempt", "risk to data principals"]
    },
    "RIGHTS_IMPLEMENTATION_VIOLATION": {
        "primary": ["right to", "erasure", "correction", "access", "portability", "object to", "unspecified means", "nominate"],
        "secondary": ["exercise your rights", "subject to change", "not be prominently published"]
    }
}