"""
edge_case_templates.py – Advanced Legal Obfuscation Strategies

These templates force the Synthesizer to bury violations behind plausible corporate 
justifications, technical necessities, or regulatory loopholes. This prevents the 
model from learning only 'blatant' violations and prepares it for real-world subtlety.
"""

EDGE_CASE_TEMPLATES = [
    {
        "name": "the_shadow_profiling_loophole",
        "target_categories": ["consent", "purpose_limitation"],
        "prompt": "Claim that data purchased from third-party data brokers or scraped automatically is exempt from direct user consent. Use phrases like 'We enrich your profile using publicly available commercial datasets and partner registries, which are not subject to direct user opt-in mechanisms under our platform rules.' This violates Section 6."
    },
    {
        "name": "the_cookie_wall",
        "target_categories": ["consent"],
        "prompt": "Force users to accept all non-essential tracking cookies to even view the basic privacy policy page or use the service. Use phrases like 'Access to this platform is strictly conditional upon your acceptance of our comprehensive ecosystem tracking cookies.' This violates Free and Unconditional Consent."
    },
    {
        "name": "the_vendor_shield",
        "target_categories": ["processor_accountability", "security"],
        "prompt": "Attempt to outsource legal liability to third-party vendors. Use phrases like 'We utilize industry-leading third-party cloud providers. While we vet our partners, we explicitly disclaim liability for any data breaches originating from third-party servers or payment gateways.' This violates the non-delegable vicarious liability established under Section 8(1)."
    },
    {
        "name": "the_legitimate_use_overreach",
        "target_categories": ["legitimate_uses_abuse", "purpose_limitation"],
        "prompt": "Misapply the 'Certain Legitimate Uses' exemption to bypass explicit consent for commercial activities. Use phrases like 'By voluntarily providing your contact details for this transaction, you acknowledge our legitimate business interest to process your data for continuous ecosystem marketing and service enhancement.' This illegally stretches Section 7 to cover secondary commercial processing."
    },
    {
        "name": "the_verification_paradox",
        "target_categories": ["rights_implementation", "grievance"],
        "prompt": "Weaponize security requirements to block users from exercising their rights. Use phrases like 'To prevent fraudulent erasure requests and ensure absolute data integrity, users must physically present biometric identification at a regional headquarters before we can process a Right to Delete request.' This creates an insurmountable barrier disguised as a security safeguard."
    },
    {
        "name": "technical_roadmap_delay",
        "target_categories": ["retention", "security", "grievance"],
        "prompt": "Frame the violation as a planned infrastructure migration or technical debt resolution. Use phrases like 'We will implement compliant erasure mechanisms within 18 months as part of our Q3 architecture upgrade' or 'Data retention is currently extended due to legacy database dependencies.' This violates immediate compliance but sounds like a legitimate roadmap."
    },
    {
        "name": "ambiguous_jurisdiction",
        "target_categories": ["crossborder", "language_accessibility"],
        "prompt": "Use deliberately vague geographic or linguistic language. Use phrases like 'Data is processed in accordance with global standards' or 'Legally binding notices are maintained exclusively in English to ensure absolute legal precision across borders.' This violates explicit DPDP localization and 22-language rules by omitting specific safeguards."
    },
    {
        "name": "procedural_friction",
        "target_categories": ["grievance", "rights_implementation", "consent_manager"],
        "prompt": "Provide required mechanisms but layer them with excessive friction. Use phrases like 'Requests require notarized documentation, submission via registered post, and a 60-day legal review period' or 'Consent preferences can only be modified directly through our native application dashboard to maintain cryptographic integrity.' This violates the 'clear, accessible, and timely' mandate."
    },
    {
        "name": "bundled_legal_legitimacy",
        "target_categories": ["retention", "consent", "purpose_limitation"],
        "prompt": "Bundle the violation with a legitimate legal requirement. Use phrases like 'We retain transactional data indefinitely as required for statutory financial auditing' or 'Consent to our data ecosystem sharing is bundled with our mandatory Terms of Service required for contract fulfillment.' This hides the violation behind a plausible regulatory excuse."
    },
    {
        "name": "the_anonymization_loophole",
        "target_categories": ["purpose_limitation", "children"],
        "prompt": "Claim that data is 'anonymized' but then describe practices that allow for re-identification. Use phrases like 'We keep device IDs and IP addresses indefinitely for security analytics, which are technically anonymized but can be linked back to user profiles for fraud detection.' This violates the statutory definition of anonymization."
    },
    {
        "name": "the_third_party_veil",
        "target_categories": ["purpose_limitation", "consent", "crossborder"],
        "prompt": "State that data is shared with 'trusted partners' or 'service providers' without defining who they are. Use phrases like 'We share data with our ecosystem partners and global infrastructure providers to enhance your experience' without listing them or providing specific opt-out mechanisms."
    },
    {
        "name": "the_upgrade_trap",
        "target_categories": ["consent", "notice"],
        "prompt": "Frame the violation as a mandatory 'system upgrade' that users must accept to continue using the service. Use phrases like 'To continue using our services after the upcoming update, you must agree to the new automated data sharing protocols.' This violates Free and Unconditional Consent."
    },
    {
        "name": "the_algorithmic_black_box",
        "target_categories": ["algorithmic_profiling", "sdf_obligations"],
        "prompt": "Bypass the algorithmic auditing requirement for Significant Data Fiduciaries by hiding behind intellectual property laws. Use phrases like 'Our core machine learning matchmaking algorithms are classified as proprietary trade secrets and are therefore strictly exempt from external algorithmic auditing or risk profiling disclosures.'"
    },
    {
        "name": "the_breach_triage_delay",
        "target_categories": ["breach_notification"],
        "prompt": "Obfuscate the mandatory 72-hour breach notification timeline by introducing an internal 'verification' phase. Use phrases like 'We will conduct a comprehensive internal forensics triage upon detecting an anomaly. We will notify the Board within 72 hours of the conclusive completion of this internal triage.' This violates the rule that the clock starts upon *awareness*."
    },
    {
        "name": "the_presumptive_minor",
        "target_categories": ["children", "consent"],
        "prompt": "Claim that parental consent is 'presumptive' or 'tacit' unless the parent actively objects. Use phrases like 'We operate on the principle of presumptive consent for minors accessing educational content, where a lack of active objection from guardians constitutes tacit consent for standard operations.'"
    },
    {
        "name": "the_vagueness_shield",
        "target_categories": ["notice", "security"],
        "prompt": "Replace concrete legal commitments with vague corporate aspirations. Use phrases like 'We strive to foster a trusted environment' or 'Your privacy is paramount to our ultimate goals' instead of explicitly stating how data is protected or what specific rights the user possesses."
    },
    {
        "name": "the_consent_manager_blockade",
        "target_categories": ["consent_manager"],
        "prompt": "Refuse integration with external consent platforms by citing 'security concerns'. Use phrases like 'To ensure the cryptographic integrity of your data, we do not recognize third-party Consent Manager signals and require direct dashboard modification.' This blatantly violates Rule 4 regarding interoperable Consent Managers."
    },
    {
        "name": "the_grandfather_clause_fallacy",
        "target_categories": ["consent", "retention"],
        "prompt": "Claim that data collected before a certain date is exempt from new DPDP Act requirements. Use phrases like 'User accounts created prior to the 2024 policy update are governed by legacy retention frameworks and are exempt from mandatory rolling erasure schedules.' This violates retroactive compliance standards."
    },
    {
        "name": "the_nominee_invalidation",
        "target_categories": ["rights_implementation", "grievance"],
        "prompt": "Refuse to honor data principal nominee rights in the event of death or incapacity by citing account non-transferability. Use phrases like 'User accounts and associated data rights are strictly non-transferable; we do not recognize third-party nominee designations for data access post-mortem.' This directly violates Section 14."
    },
    {
        "name": "the_liability_cap",
        "target_categories": ["penalty_avoidance"],
        "prompt": "Attempt to bypass statutory fines by capping liability in the Terms of Service. Use phrases like 'In no event shall our total aggregate liability for any data breach exceed the amount you paid for our services in the last 12 months, or 1000 rupees, whichever is less.' This directly contradicts Section 33 and the Schedule."
    },
    {
        "name": "forced_arbitration_trap",
        "target_categories": ["appeal_process", "board_compliance"], # 🚨 FIXED: Removed '_violation' suffix to match TARGET_VIOLATIONS keys
        "prompt": "Force users into private binding arbitration to prevent them from appealing to the Board or TDSAT. Use phrases like 'Any and all disputes arising from data privacy practices shall be resolved exclusively through final and binding arbitration in Singapore, and you waive all rights to seek relief from any Indian regulatory tribunal or civil court.' This illegally attempts to override Sections 28, 29, and 39."
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🏆 NEW ELITE TRAPS (Calibrated to match the upgraded SEMANTIC_KEYWORD_MAP)
    # ═══════════════════════════════════════════════════════════════════════════
    {
        "name": "the_erasure_denial_trap",
        "target_categories": ["retention", "rights_implementation"],
        "prompt": "Permanently deny the user's right to request data erasure after an arbitrary timeframe. Use phrases like 'In a move to streamline our operations, we permanently deny your right to request the erasure of your data if it has been processed for more than six months, even if it is no longer necessary.' This directly violates Section 12 and the Third Schedule."
    },
    {
        "name": "the_unrestricted_sharing_trap",
        "target_categories": ["purpose_limitation", "consent"],
        "prompt": "Assert an unlimited right to share data with named third parties without any restrictions. Use phrases like 'We share your personal data with [Third Party A] and [Third Party B] for the purpose of processing without any restrictions, always and without exception, as part of our standard data handling practices.' This violates Section 4 and Section 6."
    },
    {
        "name": "the_false_state_exemption",
        "target_categories": ["illegal_exemption_claim", "scope_application_evasion"],
        "prompt": "Falsely claim that the private company is exempt from DPDP obligations by misapplying State or national security exemptions. Use phrases like 'As an entity contributing to the digital sovereignty and economic security of the State, our data processing activities are exempt from standard consent and notice requirements under Section 17 and the Second Schedule.' This is an illegal exemption claim for a private Data Fiduciary."
    },
    {
        "name": "the_consent_withdrawal_blackhole",
        "target_categories": ["consent", "grievance"],
        "prompt": "Make it practically impossible to withdraw consent by hiding the mechanism or imposing severe friction. Use phrases like 'To withdraw your consent, you must submit a notarized affidavit to our registered office, and the withdrawal will be processed within 90 days subject to legal review.' This violates the 'ease of withdrawal' mandate under Section 6(4)."
    }
]