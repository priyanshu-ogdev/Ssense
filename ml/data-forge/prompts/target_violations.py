"""
target_violations.py – DPDP Act 2023 & Rules 2025 Violation Matrix
Contains pristine, atomic statutory text used as Semantic Proxy Queries 
for the BGE Vector DB RAG retrieval engine.
"""

TARGET_VIOLATIONS = {
    "purpose_limitation": [
        "Section 4(1): A person may process the personal data of a Data Principal only in accordance with the provisions of this Act and for a lawful purpose, (a) for which the Data Principal has given her consent; or (b) for certain legitimate uses.",
        "Section 4(2): For the purposes of this section, the expression “lawful purpose” means any purpose which is not expressly forbidden by law."
    ],
    "consent": [
        "Section 6(1): The consent given by the Data Principal shall be free, specific, informed, unconditional and unambiguous with a clear affirmative action, and shall signify an agreement to the processing of her personal data for the specified purpose and be limited to such personal data as is necessary for such specified purpose.",
        "Section 6(2): Any part of consent referred in sub-section (1) which constitutes an infringement of the provisions of this Act or the rules made thereunder or any other law for the time being in force shall be invalid to the extent of such infringement.",
        "Section 6(4): Where consent given by the Data Principal is the basis of processing of personal data, such Data Principal shall have the right to withdraw her consent at any time, with the ease of doing so being comparable to the ease with which such consent was given."
    ],
    "legitimate_uses_abuse": [
        "Section 7(a): A Data Fiduciary may process personal data of a Data Principal for the specified purpose for which the Data Principal has voluntarily provided her personal data to the Data Fiduciary, and in respect of which she has not indicated to the Data Fiduciary that she does not consent to the use of her personal data."
    ],
    "notice": [
        "Section 5(1): Every request made to a Data Principal under section 6 for consent shall be accompanied or preceded by a notice given by the Data Fiduciary to the Data Principal, informing her, (i) the personal data and the purpose for which the same is proposed to be processed; (ii) the manner in which she may exercise her rights under sub-section (4) of section 6 and section 13; and (iii) the manner in which the Data Principal may make a complaint to the Board.",
        "Rule 3(b): The notice shall give, in clear and plain language, a fair account of the details necessary to enable the Data Principal to give specific and informed consent for the processing of her personal data, which shall include, at the minimum, (i) an itemised description of such personal data; and (ii) the specified purpose or purposes of, and specific description of the goods or services to be provided or uses to be enabled by, such processing."
    ],
    "retention": [
        "Section 8(7): A Data Fiduciary shall, unless retention is necessary for compliance with any law for the time being in force, (a) erase personal data, upon the Data Principal withdrawing her consent or as soon as it is reasonable to assume that the specified purpose is no longer being served, whichever is earlier; and (b) cause its Data Processor to erase any personal data that was made available by the Data Fiduciary for processing to such Data Processor.",
        "Rule 8(1): A Data Fiduciary, who is of such class and is processing personal data for such corresponding purposes as are specified in Third Schedule, shall erase such personal data, unless its retention is necessary for compliance with any law for the time being in force, or, for the corresponding time period specified in the Third Schedule, if the Data Principal neither approaches such Data Fiduciary for the performance of the specified purpose nor exercises her rights in relation to such processing."
    ],
    "children": [
        "Section 9(1): The Data Fiduciary shall, before processing any personal data of a child or a person with disability who has a lawful guardian obtain verifiable consent of the parent of such child or the lawful guardian, as the case may be, in such manner as may be prescribed.",
        "Rule 10(1): A Data Fiduciary shall adopt appropriate technical and organisational measures to ensure that verifiable consent of the parent is obtained before the processing of any personal data of a child and shall observe due diligence, for checking that the individual identifying herself as the parent is an adult who is identifiable."
    ],
    "security": [
        "Section 8(5): A Data Fiduciary shall protect personal data in its possession or under its control, including in respect of any processing undertaken by it or on its behalf by a Data Processor, by taking reasonable security safeguards to prevent personal data breach.",
        "Rule 6(1): A Data Fiduciary shall protect personal data by taking reasonable security safeguards to prevent personal data breach, which shall include, at the minimum, (a) appropriate data security measures, such as securing of personal data through encryption, obfuscation, masking or the use of virtual tokens mapped to that personal data."
    ],
    "breach_notification": [
        "Section 8(6): In the event of a personal data breach, the Data Fiduciary shall give the Board and each affected Data Principal, intimation of such breach in such form and manner as may be prescribed.",
        "Rule 7(2): On becoming aware of any personal data breach, the Data Fiduciary shall intimate to the Board, (a) without delay, a description of the breach, including its nature, extent, timing and location of occurrence and the likely impact; (b) within seventy-two hours of becoming aware of the breach, updated and detailed information in respect of such description."
    ],
    "processor_accountability": [
        "Section 8(1): A Data Fiduciary shall, irrespective of any agreement to the contrary or failure of a Data Principal to carry out the duties provided under this Act, be responsible for complying with the provisions of this Act and the rules made thereunder in respect of any processing undertaken by it or on its behalf by a Data Processor.",
        "Section 8(2): A Data Fiduciary may engage, appoint, use or otherwise involve a Data Processor to process personal data on its behalf for any activity related to offering of goods or services to Data Principals only under a valid contract."
    ],
    "grievance": [
        "Section 13(1): A Data Principal shall have the right to have readily available means of grievance redressal provided by a Data Fiduciary or Consent Manager in respect of any act or omission of such Data Fiduciary or Consent Manager regarding the performance of its obligations in relation to the personal data of such Data Principal or the exercise of her rights under the provisions of this Act and the rules made thereunder.",
        "Rule 9: Every Data Fiduciary shall prominently publish on its website or app, and mention in every response to a communication for the exercise of the rights of a Data Principal under the Act, the business contact information of the Data Protection Officer, if applicable, or a person who is able to answer on behalf of the Data Fiduciary the questions of the Data Principal about the processing of her personal data."
    ],
    "sdf_obligations": [
        "Section 10(2): The Significant Data Fiduciary shall (a) appoint a Data Protection Officer who shall represent the Significant Data Fiduciary under the provisions of this Act; be based in India; and be the point of contact for the grievance redressal mechanism; (b) appoint an independent data auditor to carry out data audit; and (c) undertake periodic Data Protection Impact Assessment and periodic audit.",
        "Rule 13(1): A Significant Data Fiduciary shall, once in every period of twelve months from the date on which it is notified as such or is included in the class of Data Fiduciaries notified as such, undertake a Data Protection Impact Assessment and an audit to ensure effective observance of the provisions of this Act and the rules made thereunder."
    ],
    "algorithmic_profiling": [
        "Rule 13(3): A Significant Data Fiduciary shall observe due diligence to verify that technical measures including algorithmic software adopted by it for hosting, display, uploading, modification, publishing, transmission, storage, updating or sharing of personal data processed by it are not likely to pose a risk to the rights of Data Principals."
    ],
    "crossborder": [
        "Section 16(1): The Central Government may, by notification, restrict the transfer of personal data by a Data Fiduciary for processing to such country or territory outside India as may be so notified.",
        "Rule 15: Any personal data processed by a Data Fiduciary under the Act may be transferred outside the territory of India subject to the restriction that the Data Fiduciary shall meet such requirements as the Central Government may, by general or special order, specify in respect of making such personal data available to any foreign State, or to any person or entity under the control of or any agency of such a State."
    ],
    "consent_manager": [
        "Section 6(7): The Data Principal may give, manage, review or withdraw her consent to the Data Fiduciary through a Consent Manager.",
        "Rule 4 & First Schedule Part B: A person who fulfils the conditions for registration of Consent Managers set out in Part A of First Schedule may apply to the Board for registration as a Consent Manager. The Consent Manager shall enable a Data Principal using its platform to give consent to the processing of her personal data by a Data Fiduciary onboarded onto such platform."
    ],
    "language_accessibility": [
        "Section 5(3): The Data Fiduciary shall give the Data Principal the option to access the contents of the notice referred to in sub-sections (1) and (2) in English or any language specified in the Eighth Schedule to the Constitution.",
        "Rule 3(a): The notice given by the Data Fiduciary to the Data Principal shall be presented and be understandable independently of any other information that has been, is or may be made available by such Data Fiduciary."
    ],
    "rights_implementation": [
        "Section 11(1): The Data Principal shall have the right to obtain from the Data Fiduciary to whom she has previously given consent, for processing of personal data, upon making to it a request in such manner as may be prescribed, a summary of personal data which is being processed by such Data Fiduciary and the processing activities undertaken by that Data Fiduciary with respect to such personal data.",
        "Section 12(1): A Data Principal shall have the right to correction, completion, updating and erasure of her personal data for the processing of which she has previously given consent, in accordance with any requirement or procedure under any law for the time being in force.",
        "Section 14(1): A Data Principal shall have the right to nominate, in such manner as may be prescribed, any other individual, who shall, in the event of death or incapacity of the Data Principal, exercise the rights of the Data Principal in accordance with the provisions of this Act and the rules made thereunder."
    ],
    "board_compliance": [
        "Section 28(7): For the purposes of discharging its functions under this Act, the Board shall have the same powers as are vested in a civil court under the Code of Civil Procedure, 1908, in respect of matters relating to (a) summoning and enforcing the attendance of any person and examining her on oath; (b) receiving evidence of affidavit requiring the discovery and production of documents; (c) inspecting any data, book, document, register, books of account or any other document."
    ],
    "penalty_avoidance": [
        "Section 33(1): If the Board determines on conclusion of an inquiry that breach of the provisions of this Act or the rules made thereunder by a person is significant, it may, after giving the person an opportunity of being heard, impose such monetary penalty specified in the Schedule."
    ],
    "appeal_process": [
        "Section 29(1): Any person aggrieved by an order or direction made by the Board under this Act may prefer an appeal before the Appellate Tribunal."
    ],
    "scope_application_evasion": [
        "Section 3(b): The Act shall also apply to processing of digital personal data outside the territory of India, if such processing is in connection with any activity related to offering of goods or services to Data Principals within the territory of India."
    ],
    "illegal_exemption_claim": [
        "Section 17(2)(b): The provisions of this Act shall not apply in respect of the processing of personal data necessary for research, archiving or statistical purposes if the personal data is not to be used to take any decision specific to a Data Principal and such processing is carried on in accordance with such standards as may be prescribed."
    ]
}

# ═══════════════════════════════════════════════════════════════════════════
# 2. ATOMIC STATUTES (Prevents "Generic Statute" Validator Drops)
# ═══════════════════════════════════════════════════════════════════════════
# Includes ALL Sections, Rules, and Schedules to prevent false drops when 
# the LLM cites the parent statute rather than the specific subsection.
ATOMIC_STATUTES = [
    # All Sections (1 to 44)
    "section 1", "section 2", "section 3", "section 4", "section 5", "section 6", "section 7", 
    "section 8", "section 9", "section 10", "section 11", "section 12", "section 13", "section 14", 
    "section 15", "section 16", "section 17", "section 18", "section 19", "section 20", "section 21", 
    "section 22", "section 23", "section 24", "section 25", "section 26", "section 27", "section 28", 
    "section 29", "section 30", "section 31", "section 32", "section 33", "section 34", "section 35", 
    "section 36", "section 37", "section 38", "section 39", "section 40", "section 41", "section 42", 
    "section 43", "section 44",
    # All Rules (1 to 23)
    "rule 1", "rule 2", "rule 3", "rule 4", "rule 5", "rule 6", "rule 7", "rule 8", "rule 9", 
    "rule 10", "rule 11", "rule 12", "rule 13", "rule 14", "rule 15", "rule 16", "rule 17", 
    "rule 18", "rule 19", "rule 20", "rule 21", "rule 22", "rule 23",
    # All Schedules
    "first schedule", "second schedule", "third schedule", "fourth schedule", "fifth schedule", 
    "sixth schedule", "seventh schedule", "eighth schedule"
]

# ═══════════════════════════════════════════════════════════════════════════
# 3. SEMANTIC KEYWORD MAP (Prevents "Semantic Mismatch" Validator Drops)
# ═══════════════════════════════════════════════════════════════════════════
# Calibrated to catch both direct legal violations AND the specific corporate 
# vagueness/obfuscation traps generated by the 72B Synthesizer.
SEMANTIC_KEYWORD_MAP = {
    "PURPOSE_LIMITATION_VIOLATION": {
        "primary": ["purpose", "lawful", "specified", "collect various", "engage with", "ecosystem", "business improvement", "any purpose", "internal analytics", "machine learning", "enriches", "profile", "combining", "datasets", "partner registries", "shares", "without any restrictions", "without exception", "always", "processing", "commercially available"],
        "secondary": ["deem appropriate", "future services", "enhance your experience", "trusted partners", "opt-in", "preferences", "understanding"]
    },
    "CONSENT_NOT_FREE_OR_SPECIFIC": {
        "primary": ["consent", "bundl", "opt-out", "deemed", "manage", "submit", "continued use", "engagement", "acknowledge", "implicit", "presumptive", "unconditional", "withdraw", "cornerstone"],
        "secondary": ["integral to the service", "condition of use", "by using our", "tacit", "freely given", "specific, informed"]
    },
    "LEGITIMATE_USES_ABUSE": {
        "primary": ["legitimate use", "deemed consent", "voluntarily provided", "without explicit consent", "implicit exception", "standard business operations", "certain legitimate uses"],
        "secondary": ["medical emergency", "employment", "service improvement", "implied", "subsidy", "benefit"]
    },
    "NOTICE_INADEQUATE": {
        "primary": ["notice", "inform", "disclose", "transparent", "privacy policy", "endeavor", "strive", "paramount", "commit", "assure", "believe", "delineate", "outline", "collect", "gather", "process", "valued user", "accompany", "precede"],
        "secondary": ["clear", "prominent", "refer to", "read in conjunction", "traverse", "in strict adherence", "meticulously aligned", "trusted environment", "legal precision", "itemised", "fair account"]
    },
    "DATA_RETENTION_LIMIT_EXCEEDED": {
        "primary": ["retain", "retention", "erase", "delete", "indefinitely", "period", "as long as", "obligations", "business continuity", "legacy", "store", "maintain", "keep", "preserve", "retains", "analytics", "enhance user experience", "permanently deny", "streamline", "six months", "erasure", "minimizing", "service continuity"],
        "secondary": ["archive", "historical", "audit", "tax compliance", "statutory", "temporal limitation", "exceeds"]
    },
    "CHILD_CONSENT_VIOLATION": {
        "primary": ["child", "children", "parental", "minor", "under 18", "under eighteen", "guardian", "verifiable", "youth", "age self-certif", "presumptive", "tacit", "applies to all", "when applicable", "exigency", "best interest", "verifiable consent"],
        "secondary": ["educational", "safety", "financial literacy", "well-being", "detrimental", "tracking", "behavioural monitoring"]
    },
    "SECURITY_SAFEGUARDS_MISSING": {
        "primary": ["security", "safeguard", "encryption", "protect", "integrity", "robust", "industry-standard", "appropriate measures", "securely", "forensics", "obfuscation", "masking", "virtual tokens"],
        "secondary": ["technical", "organizational", "selective", "evolving nature", "cyber threats", "not necessarily fixed periodicity", "unauthorized access"]
    },
    "BREACH_NOTIFICATION_FAILURE": {
        "primary": ["breach", "notify", "notification", "incident", "72 hours", "triage", "verification", "without delay", "awareness", "acknowledge", "comprehensive internal", "verification phase", "intimation", "seventy-two hours"],
        "secondary": ["unauthorized", "disclosure", "forensics", "conclusive", "completion", "mitigate", "remedial"]
    },
    "PROCESSOR_ACCOUNTABILITY_VIOLATION": {
        "primary": ["disclaim liability", "not responsible for third", "vendor is solely", "out of our control", "not liable", "cannot guarantee third-party", "irrespective", "agreement to the contrary"],
        "secondary": ["payment gateway", "cloud provider", "service provider", "analytics partner", "hosting provider", "valid contract"]
    },
    "GRIEVANCE_REDRESSAL_INADEQUATE": {
        "primary": ["grievance", "redressal", "officer", "dpo", "complaint", "query", "concern", "resolve", "endeavor", "aim to", "promptly", "notarized", "registered post", "reasonable time", "forensics", "contact information", "business contact"],
        "secondary": ["contact", "days", "fax number", "written form", "customer identification file", "legitimacy", "provision", "readily available"]
    },
    "SDF_OBLIGATIONS_MISSING": {
        "primary": ["significant", "sdf", "dpo", "impact assessment", "proprietary", "trade secret", "algorithmic", "machine learning", "comprehensive internal", "compliance is at the heart", "data auditor", "periodic audit"],
        "secondary": ["audit", "million", "crore", "exempt", "in lieu of", "matchmaking algorithms", "twelve months"]
    },
    "CROSS_BORDER_TRANSFER_VIOLATION": {
        "primary": ["transfer", "outside", "foreign", "global", "international", "jurisdiction", "cross-border", "unrestricted", "reserve the right", "controlled transfers", "global infrastructure", "international best practices", "restrictions and safeguards"],
        "secondary": ["servers", "best practices", "controlled", "territory of India"]
    },
    "CONSENT_MANAGER_OBSTRUCTION": {
        "primary": ["consent manager", "cryptographic integrity", "native application", "do not recognize", "third-party signals", "proprietary dashboard", "board-registered", "interoperable platform"],
        "secondary": ["interoperable", "board-registered", "directly through our", "facilitate"]
    },
    "LANGUAGE_ACCESSIBILITY": {
        "primary": ["language", "english", "22 languages", "legally binding", "maintained exclusively", "legal precision", "eighth schedule", "clear and plain"],
        "secondary": ["regional", "translation", "prominently displayed", "independently"]
    },
    "ALGORITHMIC_PROFILING_SDF": {
        "primary": ["algorithmic", "profiling", "automated decision", "trade secrets", "machine learning", "due diligence", "audit", "algorithm", "algorithms", "proprietary", "exempting", "disclosing", "logic", "workings", "guise", "optimizing"],
        "secondary": ["black box", "proprietary", "exempt", "risk to data principals", "transparency"]
    },
    "RIGHTS_IMPLEMENTATION_VIOLATION": {
        "primary": ["right to", "erasure", "correction", "access", "portability", "object to", "unspecified means", "nominate", "permanently deny", "streamline", "six months", "nominee", "incapacity"],
        "secondary": ["exercise your rights", "subject to change", "not be prominently published", "death", "unsoundness of mind"]
    },
    "BOARD_COMPLIANCE_VIOLATION": {
        "primary": ["board", "summon", "inquiry", "inspect", "interfere", "cooperate", "defy", "withhold", "civil court", "evidence"],
        "secondary": ["investigation", "proceedings", "data protection board", "directions"]
    },
    "PENALTY_AVOIDANCE": {
        "primary": ["penalty", "fine", "cap", "liability", "rupees", "crore", "maximum", "limit", "damage", "indemnify", "monetary penalty", "two hundred and fifty"],
        "secondary": ["section 33", "schedule", "compensation", "consolidated fund"]
    },
    "APPEAL_PROCESS_VIOLATION": {
        "primary": ["appeal", "tdsat", "tribunal", "appellate", "civil court", "jurisdiction", "arbitration", "binding", "waive", "class action", "appellate tribunal", "sixty days"],
        "secondary": ["section 29", "section 30", "order", "telecom regulatory"]
    },
    "SCOPE_APPLICATION_EVASION": {
        "primary": ["scope", "territory", "offline", "digitize", "outside india", "apply", "jurisdiction", "exclude", "not covered", "territory of India", "digital form"],
        "secondary": ["section 3", "processing", "exempt", "domestic purpose"]
    },
    "ILLEGAL_EXEMPTION_CLAIM": {
        "primary": ["exemption", "exempt", "sovereignty", "security of state", "section 17", "instrumentality", "government", "research", "archiving", "statistical purposes"],
        "secondary": ["fraud", "national security", "public order", "enforcing any legal right"]
    }
}