"""Synthetic incident generator for the Agentic Incident Postmortem Synthesizer.

Emits 10+ incidents across deploys/metrics/logs/chat. Every incident carries
ground-truth fields `true_root_cause` (a `root_cause_label` slug) and `red_herring`
(a slug) so the eval is label-vs-truth, not judgement-based.

The first 3 incidents double as pre-seeded `incident_memory` docs (consulted hypotheses).

Run:  python generate_incidents.py            # writes ./incidents/*.json + ./incidents/memory_seed.json
      python generate_incidents.py --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

# Each incident: id, window, description, ground-truth slugs, and evidence list.
# Evidence ids are E1..En, scoped per incident. `ts` is ISO-8601 text.
INCIDENTS: List[Dict[str, Any]] = [
    {
        "id": "INC-001",
        "window_start": "2026-08-20T14:00:00",
        "window_end": "2026-08-20T15:00:00",
        "description": "Checkout failures and elevated payment errors after the 14:02 payment-service deploy.",
        "true_root_cause": "config_timeout_drop",
        "red_herring": "db_cpu_spike",
        "evidence": [
            {"id": "E1", "ts": "2026-08-20T14:02:00", "source": "deploys",
             "source_url": "https://ci.example/payment/142",
             "content": "payment-service v2.3.1 deployed; config change payment_timeout_ms 2000 -> 200"},
            {"id": "E2", "ts": "2026-08-20T14:03:30", "source": "metrics",
             "source_url": "https://grafana.example/checkout/latency",
             "content": "checkout p99 latency rose 800ms -> 4.2s"},
            {"id": "E3", "ts": "2026-08-20T14:10:00", "source": "metrics",
             "source_url": "https://grafana.example/checkout/errors",
             "content": "checkout error rate 0.1% -> 5.3%"},
            {"id": "E4", "ts": "2026-08-20T14:12:00", "source": "logs",
             "source_url": "https://logs.example/payment",
             "content": "PaymentGateway timeout after 200ms (expected >=2000ms)"},
            {"id": "E5", "ts": "2026-08-20T14:25:00", "source": "metrics",
             "source_url": "https://grafana.example/db/cpu",
             "content": "DB CPU spike to 95% (coincidental, unrelated batch job)"},
            {"id": "E6", "ts": "2026-08-20T14:30:00", "source": "chat",
             "source_url": "https://chat.example/inc-001",
             "content": "on-call: 'looks like the 14:02 deploy changed a timeout; DB spike is a separate batch'"},
            {"id": "E7", "ts": "2026-08-20T14:45:00", "source": "deploys",
             "source_url": "https://ci.example/payment/143",
             "content": "rollback payment_timeout_ms to 2000"},
            {"id": "E8", "ts": "2026-08-20T14:50:00", "source": "metrics",
             "source_url": "https://grafana.example/checkout/errors",
             "content": "checkout error rate recovered to 0.2%"},
        ],
    },
    {
        "id": "INC-002",
        "window_start": "2026-08-21T09:00:00",
        "window_end": "2026-08-21T10:00:00",
        "description": "API latency spikes and cache hit-rate collapse after redis node maintenance.",
        "true_root_cause": "redis_cache_eviction",
        "red_herring": "network_latency",
        "evidence": [
            {"id": "E1", "ts": "2026-08-21T09:05:00", "source": "deploys",
             "source_url": "https://ci.example/infra/77",
             "content": "redis maxmemory lowered 8GB -> 2GB during maintenance"},
            {"id": "E2", "ts": "2026-08-21T09:06:00", "source": "metrics",
             "source_url": "https://grafana.example/cache/hitrate",
             "content": "cache hit-rate 96% -> 41%"},
            {"id": "E3", "ts": "2026-08-21T09:10:00", "source": "metrics",
             "source_url": "https://grafana.example/api/latency",
             "content": "API p95 latency 120ms -> 900ms"},
            {"id": "E4", "ts": "2026-08-21T09:20:00", "source": "logs",
             "source_url": "https://logs.example/redis",
             "content": "redis: evicted 1.2M keys/sec (maxmemory reached)"},
            {"id": "E5", "ts": "2026-08-21T09:30:00", "source": "metrics",
             "source_url": "https://grafana.example/net/rtt",
             "content": "inter-AZ network RTT 2ms -> 3ms (within normal envelope)"},
            {"id": "E6", "ts": "2026-08-21T09:40:00", "source": "chat",
             "source_url": "https://chat.example/inc-002",
             "content": "on-call: 'network looks fine; redis is evicting like crazy'"},
        ],
    },
    {
        "id": "INC-003",
        "window_start": "2026-08-22T18:00:00",
        "window_end": "2026-08-22T19:00:00",
        "description": "Order processing stalled; consumers lagging after a Kafka partition rebalance.",
        "true_root_cause": "kafka_consumer_lag",
        "red_herring": "disk_io_saturation",
        "evidence": [
            {"id": "E1", "ts": "2026-08-22T18:02:00", "source": "deploys",
             "source_url": "https://ci.example/orders/55",
             "content": "orders-consumer v1.9 rolled out with session.timeout.ms 30s -> 5s"},
            {"id": "E2", "ts": "2026-08-22T18:05:00", "source": "metrics",
             "source_url": "https://grafana.example/kafka/lag",
             "content": "consumer lag 0 -> 4.5M messages"},
            {"id": "E3", "ts": "2026-08-22T18:15:00", "source": "logs",
             "source_url": "https://logs.example/orders",
             "content": "Rebalance failed: MemberId required for session.timeout 5s < processing time"},
            {"id": "E4", "ts": "2026-08-22T18:30:00", "source": "metrics",
             "source_url": "https://grafana.example/disk/io",
             "content": "disk I/O 30% (steady, no saturation)"},
            {"id": "E5", "ts": "2026-08-22T18:40:00", "source": "chat",
             "source_url": "https://chat.example/inc-003",
             "content": "on-call: 'disk fine, consumers keep rebalancing and falling behind'"},
        ],
    },
    {
        "id": "INC-004",
        "window_start": "2026-08-23T11:00:00",
        "window_end": "2026-08-23T12:00:00",
        "description": "Service discovery failures; hosts unable to resolve internal DNS.",
        "true_root_cause": "dns_resolution_failure",
        "red_herring": "load_balancer_flap",
        "evidence": [
            {"id": "E1", "ts": "2026-08-23T11:01:00", "source": "deploys",
             "source_url": "https://ci.example/coredns/12",
             "content": "coredns configmap updated; upstream forward zone typo introduced"},
            {"id": "E2", "ts": "2026-08-23T11:03:00", "source": "metrics",
             "source_url": "https://grafana.example/dns/errors",
             "content": "DNS NXDOMAIN rate 0 -> 22%"},
            {"id": "E3", "ts": "2026-08-23T11:10:00", "source": "logs",
             "source_url": "https://logs.example/coredns",
             "content": "coredns: SERVFAIL forwarding to bad upstream 10.0.0.999"},
            {"id": "E4", "ts": "2026-08-23T11:25:00", "source": "metrics",
             "source_url": "https://grafana.example/lb/health",
             "content": "LB backend health 100% (no flap)"},
            {"id": "E5", "ts": "2026-08-23T11:35:00", "source": "chat",
             "source_url": "https://chat.example/inc-004",
             "content": "on-call: 'LB healthy; DNS is returning servfail'"},
        ],
    },
    {
        "id": "INC-005",
        "window_start": "2026-08-24T13:00:00",
        "window_end": "2026-08-24T14:00:00",
        "description": "Database connections exhausted; app pods crash-looping.",
        "true_root_cause": "connection_pool_exhaustion",
        "red_herring": "cpu_throttling",
        "evidence": [
            {"id": "E1", "ts": "2026-08-24T13:02:00", "source": "deploys",
             "source_url": "https://ci.example/api/90",
             "content": "api v3.1 raised DB pool max 20 -> 200 per pod (x40 pods = 8000)"},
            {"id": "E2", "ts": "2026-08-24T13:04:00", "source": "metrics",
             "source_url": "https://grafana.example/db/conns",
             "content": "DB active connections hit max 6000"},
            {"id": "E3", "ts": "2026-08-24T13:12:00", "source": "logs",
             "source_url": "https://logs.example/api",
             "content": "FATAL: remaining connection slots are reserved"},
            {"id": "E4", "ts": "2026-08-24T13:30:00", "source": "metrics",
             "source_url": "https://grafana.example/cpu/throttle",
             "content": "CPU throttling 4% (nominal)"},
            {"id": "E5", "ts": "2026-08-24T13:40:00", "source": "chat",
             "source_url": "https://chat.example/inc-005",
             "content": "on-call: 'cpu fine; we just opened 8000 connections to a 6000-cap db'"},
        ],
    },
    {
        "id": "INC-006",
        "window_start": "2026-08-25T20:00:00",
        "window_end": "2026-08-25T21:00:00",
        "description": "Sudden 429s for a partner API after a config push.",
        "true_root_cause": "rate_limit_misconfig",
        "red_herring": "db_deadlock",
        "evidence": [
            {"id": "E1", "ts": "2026-08-25T20:01:00", "source": "deploys",
             "source_url": "https://ci.example/gateway/31",
             "content": "gateway config: partner rate limit 5000/min -> 50/min"},
            {"id": "E2", "ts": "2026-08-25T20:03:00", "source": "metrics",
             "source_url": "https://grafana.example/gw/429",
             "content": "HTTP 429 rate 0% -> 38%"},
            {"id": "E3", "ts": "2026-08-25T20:15:00", "source": "logs",
             "source_url": "https://logs.example/gateway",
             "content": "rate limit exceeded for partner token pid-77"},
            {"id": "E4", "ts": "2026-08-25T20:25:00", "source": "metrics",
             "source_url": "https://grafana.example/db/locks",
             "content": "DB lock wait 0ms (no deadlock)"},
            {"id": "E5", "ts": "2026-08-25T20:35:00", "source": "chat",
             "source_url": "https://chat.example/inc-006",
             "content": "on-call: 'no db locks; limit got mis-set to 50/min'"},
        ],
    },
    {
        "id": "INC-007",
        "window_start": "2026-08-26T07:00:00",
        "window_end": "2026-08-26T08:00:00",
        "description": "TLS handshake failures across services after cert rotation.",
        "true_root_cause": "cert_expiry_tls",
        "red_herring": "cdn_cache_miss",
        "evidence": [
            {"id": "E1", "ts": "2026-08-26T07:00:00", "source": "deploys",
             "source_url": "https://ci.example/certs/8",
             "content": "rotated TLS cert with wrong SAN (omitted api.internal)"},
            {"id": "E2", "ts": "2026-08-26T07:02:00", "source": "metrics",
             "source_url": "https://grafana.example/tls/fails",
             "content": "TLS handshake failures 0 -> 19%"},
            {"id": "E3", "ts": "2026-08-26T07:10:00", "source": "logs",
             "source_url": "https://logs.example/tls",
             "content": "x509: certificate is valid for web only, not api.internal"},
            {"id": "E4", "ts": "2026-08-26T07:20:00", "source": "metrics",
             "source_url": "https://grafana.example/cdn/miss",
             "content": "CDN cache miss 12% (unchanged baseline)"},
            {"id": "E5", "ts": "2026-08-26T07:30:00", "source": "chat",
             "source_url": "https://chat.example/inc-007",
             "content": "on-call: 'cert SAN missing; cdn miss rate is normal'"},
        ],
    },
    {
        "id": "INC-008",
        "window_start": "2026-08-27T16:00:00",
        "window_end": "2026-08-27T17:00:00",
        "description": "Capacity oscillation; instances repeatedly terminated and respawned.",
        "true_root_cause": "autoscaler_flap",
        "red_herring": "memory_leak",
        "evidence": [
            {"id": "E1", "ts": "2026-08-27T16:01:00", "source": "deploys",
             "source_url": "https://ci.example/autoscale/4",
             "content": "HPA metric switched cpu -> qps with threshold 5 (too low)"},
            {"id": "E2", "ts": "2026-08-27T16:05:00", "source": "metrics",
             "source_url": "https://grafana.example/hpa/events",
             "content": "replicas oscillating 3<->30 every 60s"},
            {"id": "E3", "ts": "2026-08-27T16:15:00", "source": "logs",
             "source_url": "https://logs.example/hpa",
             "content": "scale down triggered: qps 4 < 5 target"},
            {"id": "E4", "ts": "2026-08-27T16:30:00", "source": "metrics",
             "source_url": "https://grafana.example/mem/heap",
             "content": "heap stable 220MB (no leak)"},
            {"id": "E5", "ts": "2026-08-27T16:40:00", "source": "chat",
             "source_url": "https://chat.example/inc-008",
             "content": "on-call: 'mem fine; the HPA target is just wrong'"},
        ],
    },
    {
        "id": "INC-009",
        "window_start": "2026-08-28T10:00:00",
        "window_end": "2026-08-28T11:00:00",
        "description": "Feature rollout caused errors for a subset of users.",
        "true_root_cause": "feature_flag_wrong",
        "red_herring": "queue_backlog",
        "evidence": [
            {"id": "E1", "ts": "2026-08-28T10:01:00", "source": "deploys",
             "source_url": "https://ci.example/flags/19",
             "content": "flag 'new-checkout' enabled for 100% (intended 10%)"},
            {"id": "E2", "ts": "2026-08-28T10:04:00", "source": "metrics",
             "source_url": "https://grafana.example/errors/byflag",
             "content": "errors correlate 1:1 with flag=new-checkout users"},
            {"id": "E3", "ts": "2026-08-28T10:12:00", "source": "logs",
             "source_url": "https://logs.example/api",
             "content": "NullPointer when new-checkout path hit uninitialized cart"},
            {"id": "E4", "ts": "2026-08-28T10:25:00", "source": "metrics",
             "source_url": "https://grafana.example/queue/backlog",
             "content": "queue backlog 0 (empty)"},
            {"id": "E5", "ts": "2026-08-28T10:35:00", "source": "chat",
             "source_url": "https://chat.example/inc-009",
             "content": "on-call: 'queue empty; the flag rolled to everyone'"},
        ],
    },
    {
        "id": "INC-010",
        "window_start": "2026-08-29T12:00:00",
        "window_end": "2026-08-29T13:00:00",
        "description": "Upstream gRPC calls timing out after an upstream service upgrade.",
        "true_root_cause": "grpc_upstream_timeout",
        "red_herring": "node_oom",
        "evidence": [
            {"id": "E1", "ts": "2026-08-29T12:01:00", "source": "deploys",
             "source_url": "https://ci.example/upstream/3",
             "content": "upstream inventory-svc upgraded; grpc keepalive tightened"},
            {"id": "E2", "ts": "2026-08-29T12:03:00", "source": "metrics",
             "source_url": "https://grafana.example/grpc/timeout",
             "content": "grpc deadline_exceeded 0 -> 17%"},
            {"id": "E3", "ts": "2026-08-29T12:12:00", "source": "logs",
             "source_url": "https://logs.example/grpc",
             "content": "received RST_STREAM after 100ms keepalive on inventory-svc"},
            {"id": "E4", "ts": "2026-08-29T12:25:00", "source": "metrics",
             "source_url": "https://grafana.example/node/mem",
             "content": "node memory 55% (no OOM)"},
            {"id": "E5", "ts": "2026-08-29T12:35:00", "source": "chat",
             "source_url": "https://chat.example/inc-010",
             "content": "on-call: 'no oom; upstream grpc keepalive is killing streams'"},
        ],
    },
    {
        "id": "INC-011",
        "window_start": "2026-08-30T15:00:00",
        "window_end": "2026-08-30T16:00:00",
        "description": "Image uploads failing intermittently after a storage endpoint change.",
        "true_root_cause": "s3_throttling_config",
        "red_herring": "region_outage",
        "evidence": [
            {"id": "E1", "ts": "2026-08-30T15:01:00", "source": "deploys",
             "source_url": "https://ci.example/storage/21",
             "content": "storage client request rate limit set 1000/s -> 10/s"},
            {"id": "E2", "ts": "2026-08-30T15:03:00", "source": "metrics",
             "source_url": "https://grafana.example/s3/429",
             "content": "S3 429 SlowDown 0 -> 29%"},
            {"id": "E3", "ts": "2026-08-30T15:12:00", "source": "logs",
             "source_url": "https://logs.example/storage",
             "content": "SlowDown: please reduce your request rate"},
            {"id": "E4", "ts": "2026-08-30T15:25:00", "source": "metrics",
             "source_url": "https://grafana.example/region/status",
             "content": "region health GREEN (no outage)"},
            {"id": "E5", "ts": "2026-08-30T15:35:00", "source": "chat",
             "source_url": "https://chat.example/inc-011",
             "content": "on-call: 'region is green; our client rate limit is just 10/s'"},
        ],
    },
]


def memory_doc(incident: Dict[str, Any]) -> Dict[str, Any]:
    """Build an `incident_memory` document for a single (approved) incident."""
    ev = incident["evidence"]
    symptoms = "; ".join(e["content"] for e in ev[:3])
    return {
        "incident_id": incident["id"],
        "document": (
            f"{incident['description']}\n"
            f"Symptoms: {symptoms}\n"
            f"Root cause: {incident['true_root_cause']}\n"
            f"Action items: review config change for {incident['true_root_cause']}"
        ),
        "metadata": {
            "incident_id": incident["id"],
            "root_cause_label": incident["true_root_cause"],
            "time_approved": incident["window_end"],
            "action_item_count": 1,
            "symptom_keywords": ",".join(
                incident["true_root_cause"].split("_") + incident["red_herring"].split("_")
            ),
        },
    }


def generate_incidents() -> List[Dict[str, Any]]:
    """Return the full list of synthetic incidents (with ground truth)."""
    return [dict(inc) for inc in INCIDENTS]


def memory_seed() -> List[Dict[str, Any]]:
    """First 3 incidents as pre-seeded `incident_memory` consulted hypotheses."""
    return [memory_doc(inc) for inc in INCIDENTS[:3]]


def write_incidents(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for inc in INCIDENTS:
        with open(os.path.join(out_dir, f"{inc['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(inc, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "memory_seed.json"), "w", encoding="utf-8") as f:
        json.dump(memory_seed(), f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(INCIDENTS)} incidents + memory_seed to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="incidents", help="output directory")
    args = ap.parse_args()
    write_incidents(args.out)


if __name__ == "__main__":
    main()
