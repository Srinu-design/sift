// Query load for Sift: POST /v1/query at a constant arrival rate (build guide, week 6).
// loadtest/query_bench.sh drives it; on its own:
//
//   docker run --rm --network host -v "$PWD/loadtest:/work" -w /work \
//     -e BASE=http://127.0.0.1:8200 -e KEY=sift_live_... -e RATE=60 grafana/k6:2.3.0 \
//     run k6/query.js
//
// RATE      requests per minute (default 60; per minute so rates below 1/s are possible)
// DURATION  how long (default 2m)
// SUMMARY   where to write the end-of-test summary as JSON (optional)
//
// The mix: each request picks one of the 86 dev golden questions uniformly at random, so a
// 2-minute run repeats questions (the answer cache's hit rate depends on that mix).
import http from 'k6/http';
import { check } from 'k6';

const QUESTIONS = JSON.parse(open('./questions.json'));
const RATE = Number(__ENV.RATE || 60);

export const options = {
  scenarios: {
    query: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1m',
      duration: __ENV.DURATION || '2m',
      preAllocatedVUs: Number(__ENV.VUS || 50),
      maxVUs: Number(__ENV.MAX_VUS || 400),
    },
  },
  thresholds: { http_req_failed: ['rate<0.01'], http_req_duration: ['p(95)<3000'] },
};

export default function () {
  const question = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
  const res = http.post(
    `${__ENV.BASE}/v1/query`,
    JSON.stringify({ question, collection: 'pgdocs' }),
    {
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${__ENV.KEY}` },
      timeout: '60s',
    },
  );
  check(res, { 'status is 200': (r) => r.status === 200 });
}

export function handleSummary(data) {
  return __ENV.SUMMARY ? { [__ENV.SUMMARY]: JSON.stringify(data) } : {};
}
