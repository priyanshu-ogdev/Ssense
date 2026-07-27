"""
edge_case_templates.py – Advanced Legal Obfuscation Strategies

These templates instruct the Synthesizer using pure zero-shot conceptual directives 
to encapsulate violations behind plausible corporate justifications, technical necessities, 
or regulatory loopholes without providing hardcoded text or copyable example quotes.
"""

EDGE_CASE_TEMPLATES = [
    {
        "name": "the_shadow_profiling_loophole",
        "target_categories": ["consent", "purpose_limitation"],
        "prompt": "Synthesize an unlawful operational mechanism claiming that personal data acquired from commercial data brokers or automated scrapers is permanently exempt from direct user consent, asserting an unyielding right to process such data for profile enrichment under Section 6."
    },
    {
        "name": "the_cookie_wall",
        "target_categories": ["consent"],
        "prompt": "Synthesize an unlawful operational mechanism conditioning basic service access or privacy notice viewing on mandatory consent for non-essential commercial tracking, violating the statutory mandate for free and unconditional consent under Section 6."
    },
    {
        "name": "the_vendor_shield",
        "target_categories": ["processor_accountability", "security"],
        "prompt": "Synthesize an unlawful operational mechanism explicitly disclaiming corporate liability for data breaches or processing failures occurring on third-party cloud infrastructure or payment processors, violating non-delegable fiduciary responsibility under Section 8(1)."
    },
    {
        "name": "the_legitimate_use_overreach",
        "target_categories": ["legitimate_uses_abuse", "purpose_limitation"],
        "prompt": "Synthesize an unlawful operational mechanism stretching the 'Certain Legitimate Uses' exemption to justify mandatory, un-consented processing of personal data for secondary commercial marketing or ecosystem analytics under Section 7."
    },
    {
        "name": "the_verification_paradox",
        "target_categories": ["rights_implementation", "grievance"],
        "prompt": "Synthesize an unlawful operational mechanism imposing insurmountable physical, financial, or bureaucratic identification barriers before honoring statutory data erasure or access requests under Section 12."
    },
    {
        "name": "technical_roadmap_delay",
        "target_categories": ["retention", "security", "grievance"],
        "prompt": "Synthesize an unlawful operational mechanism deferring mandatory statutory data erasure or breach safeguards to indefinite future infrastructure upgrades or technical roadmap timelines under Section 8(7)."
    },
    {
        "name": "ambiguous_jurisdiction",
        "target_categories": ["crossborder", "language_accessibility"],
        "prompt": "Synthesize an unlawful operational mechanism claiming global data transfer freedom or restricting binding legal notices exclusively to foreign languages, evading Central Government transfer restrictions under Section 16 and Eighth Schedule language requirements under Section 5."
    },
    {
        "name": "procedural_friction",
        "target_categories": ["grievance", "rights_implementation", "consent_manager"],
        "prompt": "Synthesize an unlawful operational mechanism layering extreme procedural friction (such as notarized postal submissions or mandatory 90-day legal reviews) onto consent withdrawal or grievance redressal requests under Section 6(4) and Section 13."
    },
    {
        "name": "bundled_legal_legitimacy",
        "target_categories": ["retention", "consent", "purpose_limitation"],
        "prompt": "Synthesize an unlawful operational mechanism bundling mandatory consent for secondary commercial data monetization directly with core service terms or statutory financial auditing excuses under Section 6."
    },
    {
        "name": "the_anonymization_loophole",
        "target_categories": ["purpose_limitation", "children"],
        "prompt": "Synthesize an unlawful operational mechanism asserting permanent retention of device telemetry, IP logs, or minor identifiers under the guise of pseudo-anonymization while retaining active re-identification capabilities under Section 4."
    },
    {
        "name": "the_third_party_veil",
        "target_categories": ["purpose_limitation", "consent", "crossborder"],
        "prompt": "Synthesize an unlawful operational mechanism declaring unrestricted, unconditional sharing of personal data with undefined global partners without specific purpose disclosures or opt-out controls under Section 4 and Section 6."
    },
    {
        "name": "the_upgrade_trap",
        "target_categories": ["consent", "notice"],
        "prompt": "Synthesize an unlawful operational mechanism declaring mandatory consent for expanded automated profiling as an obligatory requirement for upcoming application or service updates under Section 6."
    },
    {
        "name": "the_algorithmic_black_box",
        "target_categories": ["algorithmic_profiling", "sdf_obligations"],
        "prompt": "Synthesize an unlawful operational mechanism claiming complete exemption from mandatory periodic data audits or algorithmic risk profiling for Significant Data Fiduciaries by invoking proprietary trade secret protections under Section 10 and Rule 13."
    },
    {
        "name": "the_breach_triage_delay",
        "target_categories": ["breach_notification"],
        "prompt": "Synthesize an unlawful operational mechanism delaying mandatory 72-hour Data Protection Board breach notifications until after the indefinite completion of internal corporate forensics triage under Section 8(6) and Rule 7."
    },
    {
        "name": "the_presumptive_minor",
        "target_categories": ["children", "consent"],
        "prompt": "Synthesize an unlawful operational mechanism asserting presumptive or tacit parental consent for processing children's personal data in the absence of active guardian objections, violating verifiable parental consent rules under Section 9 and Rule 10."
    },
    {
        "name": "the_vagueness_shield",
        "target_categories": ["notice", "security"],
        "prompt": "Synthesize an unlawful operational mechanism replacing mandatory statutory notice disclosures and technical safeguard commitments with aspirational corporate statements under Section 5 and Section 8(5)."
    },
    {
        "name": "the_consent_manager_blockade",
        "target_categories": ["consent_manager"],
        "prompt": "Synthesize an unlawful operational mechanism refusing to recognize or interoperate with Board-registered Consent Managers by citing platform security or proprietary dashboard requirements under Section 6(7) and Rule 4."
    },
    {
        "name": "the_grandfather_clause_fallacy",
        "target_categories": ["consent", "retention"],
        "prompt": "Synthesize an unlawful operational mechanism claiming that user data collected prior to policy updates is permanently exempt from statutory rolling erasure schedules or DPDP consent requirements under Section 8(7)."
    },
    {
        "name": "the_nominee_invalidation",
        "target_categories": ["rights_implementation", "grievance"],
        "prompt": "Synthesize an unlawful operational mechanism declaring user accounts and data rights strictly non-transferable, actively refusing to honor statutory nominee designations post-mortem under Section 14."
    },
    {
        "name": "the_liability_cap",
        "target_categories": ["penalty_avoidance"],
        "prompt": "Synthesize an unlawful operational mechanism capping total corporate breach liability to nominal fee refunds in service terms, attempting to evade statutory monetary penalties under Section 33 and the Schedule."
    },
    {
        "name": "forced_arbitration_trap",
        "target_categories": ["appeal_process", "board_compliance"],
        "prompt": "Synthesize an unlawful operational mechanism mandating private foreign arbitration for all data privacy disputes, attempting to strip Data Principals of their statutory rights to appeal before the Board or Appellate Tribunal under Sections 28, 29, and 39."
    },
    {
        "name": "the_erasure_denial_trap",
        "target_categories": ["retention", "rights_implementation"],
        "prompt": "Synthesize an unlawful operational mechanism explicitly and permanently denying user data erasure requests for data processed past an arbitrary corporate timeframe, violating Section 12 and Rule 8."
    },
    {
        "name": "the_unrestricted_sharing_trap",
        "target_categories": ["purpose_limitation", "consent"],
        "prompt": "Synthesize an unlawful operational mechanism asserting an absolute, unrestricted right to share personal data with external third parties without purpose boundaries or consent limits under Section 4 and Section 6."
    },
    {
        "name": "the_false_state_exemption",
        "target_categories": ["illegal_exemption_claim", "scope_application_evasion"],
        "prompt": "Synthesize an unlawful operational mechanism falsely claiming that a private commercial Data Fiduciary is exempt from DPDP consent and notice requirements by misapplying sovereign State security exemptions under Section 17."
    },
    {
        "name": "the_consent_withdrawal_blackhole",
        "target_categories": ["consent", "grievance"],
        "prompt": "Synthesize an unlawful operational mechanism imposing severe friction, notarization demands, or multi-month delays on consent withdrawal requests, violating the statutory ease-of-withdrawal mandate under Section 6(4)."
    }
]