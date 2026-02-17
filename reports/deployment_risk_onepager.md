# Deployment Risk & Mitigation One-Pager

## Top Risks
- Unsafe reassurance in high-risk symptoms
- Unsupported claims without evidence
- Overconfident tone under weak factual support

## Guardrails
- High-risk red-flag detector with mandatory emergency escalation
- Citation requirement for key claims
- Rule-based safety disclaimer insertion

## Monitoring KPIs
- unsafe_advice_rate
- high_risk_miss_rate
- citation_sufficiency_rate
- overconfidence_rate
- abstention_appropriateness_rate

Current unsafe advice trend (baseline -> enhanced): 14.3% -> 4.8%
Current high-risk miss trend (baseline -> enhanced): 14.3% -> 4.8%
Current escalation compliance trend (baseline -> enhanced): 85.7% -> 95.2%

## Rollback and Human-in-the-Loop Policy
- Trigger rollback if `unsafe_advice_rate` > 20.0% or `high_risk_miss_rate` > 10.0% for 2 consecutive runs.
- Route all high-risk/ambiguous cases to human clinician review.
- Log full prompt, retrieved context, and final answer for post-incident audits.
- Maintain a versioned guardrail policy with canary release before broad rollout.

## Safety Position
System must remain advisory and non-diagnostic. High-risk decisions require escalation and human oversight.
