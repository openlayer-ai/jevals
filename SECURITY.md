# Security

jevals is a library for building guardrails, so bugs in it can turn into bugs in yours.

## Reporting

Use GitHub's private vulnerability reporting on this repository (Security tab → Report a vulnerability). Please do not open a public issue for anything that could be exploited.

We aim to acknowledge within 3 business days.

## Scope

In scope: anything that lets a gate be bypassed by input the gate is meant to evaluate (for example, a way to make `PromptInjection` or `PII` skip a question it should have asked), the safe expression evaluator in `_expr.py` executing code it should not, or the MCP server doing something beyond its listed tools.

Out of scope: a model returning a wrong probability. That is a calibration question, not a vulnerability; open an issue with the sample.

## What jevals does not do

A classifier is not an authorizer. `ToolCallRisk` can say a call looks destructive; whether it may run depends on permissions and account state that jevals cannot see. Keep a human on the irreversible actions.
