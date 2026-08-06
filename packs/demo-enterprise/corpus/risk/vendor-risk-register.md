# Vendor Risk Register — Acme Cloud Services

Maintained by Risk & Compliance. Last review: Q1. Next review: Q3.

## Classification

Acme Cloud Services is classified as a **critical third-party provider**: the
platform supports a business process whose interruption would be material within
48 hours. Critical providers are subject to enhanced due diligence, annual
on-site review, and a documented exit plan.

## Regulatory Context

The engagement falls within scope of NIS2 as an ICT service supporting an
essential entity, and processing of personal data brings it within GDPR. Because
Acme processes personal data on our instructions, the Article 28 processor
obligations apply and are reflected in clause 6 of the agreement. Data residency
is contractually confined to the European Economic Area, which satisfies the
current residency requirement.

DORA does not currently apply to this entity, but the register assumes it may on
the next regulatory review, and the exit plan is written to DORA standards for
that reason.

## Identified Risks

**R-01 · Concentration risk — severity: high.** Acme is the sole provider for
this workload with no active secondary. Migration to an alternative provider is
estimated at four to six months. The ninety-day termination-for-convenience
notice in clause 9.2 is shorter than the realistic migration window, which means
exercising it without a prepared exit leaves a coverage gap.

**R-02 · Availability risk — severity: medium.** The 99.9% monthly warranty
permits approximately 43 minutes of downtime per month without breach. Service
credits are capped at 20% of the monthly fee and are the sole remedy, so the
contractual recovery is far below the business impact of a sustained outage. The
financial remedy is therefore not a risk control; continuity planning is.

**R-03 · Data protection risk — severity: high.** A personal data breach at the
processor would engage both regulatory exposure and the extended contractual
liability under clause 7.3. The 24-hour breach notification commitment is
tighter than the regulatory 72-hour window, which is favourable, but depends on
Acme's own detection capability, which we do not audit directly.

**R-04 · Financial exposure risk — severity: medium.** The liability cap is
substantially below the plausible loss from a serious incident. Residual
exposure above the cap is uninsured at present.

## Controls in Place

- Quarterly service review with documented action tracking.
- Annual penetration test evidence requested from the supplier.
- Encrypted backups held outside the supplier's environment, tested quarterly.
- Contractual audit right, exercised once in the past twelve months.

## Recommended Additional Controls

1. Establish a warm secondary provider for the critical workload, reducing R-01
   from high to medium.
2. Extend cyber insurance to cover residual exposure above the contractual cap,
   addressing R-04.
3. Negotiate a longer termination notice or a committed transition period
   matching the realistic migration window.
4. Obtain independent assurance over the supplier's breach detection capability.

## Residual Position

With current controls, the overall residual risk of the engagement is assessed as
**high**, driven principally by concentration risk (R-01) and by the gap between
contractual liability and plausible loss (R-04).
