# Quillon Labs Engineering Onboarding

Welcome to Quillon Labs! This guide covers the first two weeks for new engineers in the Toronto office.

## Accounts and tools

Request access to GitHub, the Grafana dashboards and the staging Kubernetes cluster through the IT portal on your first day. Laptops come with Docker and Python preinstalled.

## Code review policy

Every change needs one approving review from a code owner. Changes to database migrations need a second review from the platform team. Pull requests should stay under 400 changed lines where possible.

## Deployment

Services deploy automatically after a merge to main. Each release first goes to a canary that receives 5 percent of traffic for 30 minutes; if error rates stay flat, the rollout continues to all pods.

## On-call

Engineers join the on-call rotation after their sixth week. Rotations last one week, start on Tuesday at 10:00 Toronto time, and come with a stipend of 300 dollars per week. Jonas Lindqvist runs the on-call onboarding session.

## Who to ask

- Search relevance questions: Priya Desai
- Infrastructure and on-call: Jonas Lindqvist
- Halcyon architecture: Alice Chen
- Company policies: the People Operations team
