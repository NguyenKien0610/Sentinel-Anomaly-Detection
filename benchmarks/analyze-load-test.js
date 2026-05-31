import http from "k6/http";
import { check, sleep } from "k6";
import { Rate } from "k6/metrics";

export const errorRate = new Rate("errors");

export const options = {
  scenarios: {
    steady_load: {
      executor: "constant-vus",
      vus: Number(__ENV.VUS || 20),
      duration: __ENV.DURATION || "60s",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],
    http_req_duration: ["p(95)<500", "p(99)<750"],
    errors: ["rate<0.01"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const payload = JSON.stringify({
  feature_15: 0.3187968986906816,
  feature_16: 0.4518560659934575,
  feature_19: 0.6098826098826098,
  feature_20: 0.0084317032040472,
  feature_9: 0.4470765464645514,
});

const params = {
  headers: {
    "Content-Type": "application/json",
  },
};

export default function () {
  const response = http.post(`${BASE_URL}/api/v1/analyze`, payload, params);
  const ok = check(response, {
    "status is 200": (r) => r.status === 200,
    "response has prediction": (r) => Boolean(r.json("prediction")),
    "request id exists": (r) => Boolean(r.headers["X-Request-Id"]),
  });
  errorRate.add(!ok);
  sleep(0.1);
}
