# Security policy

## Reporting a vulnerability

Please report security issues privately to **carlos.schmiedel@gmail.com** rather than opening a public
issue. Include a description, reproduction steps, and the affected version. We aim to acknowledge
within three business days.

## Deployment notes

ALM is a component, not a finished product, and a few properties are worth stating plainly.

**The API ships unauthenticated.** Setting `ALM_API_TOKEN` enables a single bearer token; that is
deliberately minimal, and the API belongs behind your own gateway. `alm serve` warns when the token
is unset.

**IBAC governs agents, not users.** It constrains what each Expert Agent may read and do inside the
federation. It is not an identity provider and does not authenticate the caller — supply
`principal` and `claims` from an authenticated upstream, and treat unauthenticated callers as
anonymous.

**Governance can be disabled.** `ALM_GOVERNANCE_ENABLED=false` allows every context read. Do not set
it in an environment holding real data.

**Model credentials are stored in the database.** API keys registered with `alm model add --api-key`
are persisted in the `models` table. They are excluded from API responses and CLI output, but the
database itself should be protected accordingly; prefer environment variables where you can.

**Domain packs are code-adjacent.** Installing a pack writes ontologies, capability declarations and
IBAC policies into your Context Graph. Review a pack before installing it, exactly as you would a
dependency — `alm pack validate` reports its contents.

**Prompt content reaches configured backends.** Retrieved context is sent to whichever backend an
expert is bound to. When residency matters, bind experts to local backends (`ollama`, `vllm`,
`transformers`) and confirm with `alm doctor` that nothing is routing to a hosted endpoint.
