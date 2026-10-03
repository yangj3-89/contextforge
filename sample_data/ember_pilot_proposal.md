# Project Ember: route optimization pilot

Sponsor: Samuel Okafor, Director of Operations Research, Kestrel Logistics
Product manager: Alice Moreno

## Why

Kestrel Logistics plans delivery routes by hand at most depots. Dispatchers spend about two hours every morning assigning stops, and late deliveries trigger contractual penalties with retail customers.

## What we will build

Project Ember computes daily routes with Google OR-Tools as a vehicle routing problem with time windows. A Python service pulls orders at 05:00, solves routes for each van, and caches traffic estimates in Redis. Dispatchers can still override any route in the dispatch app.

## Pilot

The pilot runs at the Lisbon depot with 42 vans for twelve weeks starting in October 2025. Success means raising the on-time delivery rate above 90 percent without adding vehicles.

## Risks

Driver adoption is the main risk: if drivers ignore suggested routes, the savings disappear. Alice Moreno will run weekly feedback sessions with drivers during the pilot.
