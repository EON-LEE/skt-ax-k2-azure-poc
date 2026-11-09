# shellcheck shell=bash
# Benchmark-protocol phases of aml/src/bench/slo_protocol.json (docs/BENCHMARK_PROTOCOL.md): slo-* (and the
# slo-round shorthand), diag-*, cap-*, dx1-* and dx2-*. entry.sh sources this file on every node before the
# IB probe and the downloads; it does nothing unless PHASE_PLAN names such a phase, so the demo and the
# older phases never depend on it. Uses log/report/die from entry.sh at source time and its server helpers
# (wait_healthy, server_facts, failure_tail, parallel_facts, phase_result) when a phase runs.
#
# Guards of a protocol phase, unlike plan_phase: one flag set from the protocol for every layout, a pinned
# --max-model-len and --api-server-count, no --enforce-eager retry, FP8 two-node phases only over InfiniBand
# with no NCCL communicator on NET/Socket, a clean start on every node it uses, vllm serve in its own session
# whose whole process group is stopped and checked afterwards, VLLM_SERVER_DEV_MODE=1 for this server only.
case " ${PHASE_PLAN:-} " in
  *" slo-"*|*" diag-"*|*" cap-"*|*" dx1-"*|*" dx2-"*) ;;
  *) return 0 ;;
esac
BENCH="$SRC/bench"
ROUND="${ROUND:-1}"
PHASE_PLAN=$(python3 "$BENCH/slo_suites.py" plan --round "$ROUND" $PHASE_PLAN) ||
  die 15 "PHASE_PLAN is not a plan of bench/slo_protocol.json"
BENCH_PLAN=1
log "benchmark protocol, round $ROUND: $PHASE_PLAN"

# GPU telemetry every second and InfiniBand port data counters (4-byte words) every 5 s, on every node. Both
# write only to their files (never to the job's log streams) and stop when entry.sh exits.
SMI_FIELDS=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used,clocks_event_reasons.active
nvidia-smi --query-gpu="$SMI_FIELDS" --format=csv > /dev/null 2>&1 ||
  SMI_FIELDS=${SMI_FIELDS/clocks_event_reasons/clocks_throttle_reasons}
nvidia-smi --query-gpu="$SMI_FIELDS" --format=csv -lms 1000 > "$OUT/gpu-telemetry.csv" 2>&1 < /dev/null &
BENCH_LOGGERS=$!
(
  while true; do
    line="{\"t\": $(date +%s)"
    for counters in /sys/class/infiniband/*/ports/*/counters; do
      [ -r "$counters/port_xmit_data" ] || continue
      port=${counters%/counters}; dev=${port%/ports/*}
      line+=", \"${dev##*/}/${port##*/}\": [$(cat "$counters/port_xmit_data"), $(cat "$counters/port_rcv_data")]"
    done
    echo "$line}"
    sleep 5
  done
) > "$OUT/ib-counters.jsonl" 2> /dev/null < /dev/null &
BENCH_LOGGERS="$BENCH_LOGGERS $!"
# Every NCCL process writes its own log (communicator membership and transports), read by bench_nccl. The
# directory is the same on every node: vLLM's Ray DP backend copies node0's NCCL_* variables into the engine
# actors on node1, so a per-node path would not exist there. %h keeps the nodes' files apart; on exit each node
# copies its files into its own $OUT/nccl.
BENCH_NCCL_DIR=/tmp/axk2-nccl
mkdir -p "$BENCH_NCCL_DIR" "$OUT/nccl"
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV NCCL_DEBUG_FILE="$BENCH_NCCL_DIR/%h.%p.log"
trap 'kill $BENCH_LOGGERS 2>/dev/null; cp -p "$BENCH_NCCL_DIR"/*.log "$OUT/nccl/" 2>/dev/null' EXIT

# A round job serves D (NVFP4, node0 only) next to the FP8 layouts: node0 also stages skt/A.X-K2-NVFP4,
# after the IB probe, and refuses to start when the local disk cannot hold both checkpoints.
if [ "$RANK" = 0 ] && [ -n "${NVFP4_IN:-}" ]; then
  read -r need free < <(python3 - "$BENCH/slo_protocol.json" "$PWD" "$FULL_IN" "${FULL_REPO:-skt/A.X-K2}" <<'EOF'
import json, shutil, sys
sizes = json.load(open(sys.argv[1]))["checkpoints"]
full = [c["size_gb"] for c in sizes.values() if isinstance(c, dict) and c.get("repo") == sys.argv[4]]
need = ((full[0] if full and sys.argv[3] != "none" else 0) + sizes["nvfp4"]["size_gb"]) * (1 + sizes["disk_margin_fraction"])
print(round(need + sizes["disk_margin_gb"]), round(shutil.disk_usage(sys.argv[2]).free / 1e9))
EOF
  )
  report "node0.disk_check" --text "{\"need_gb\": ${need:-null}, \"free_gb\": ${free:-null}}"
  [ -n "${need:-}" ] && [ "${free:-0}" -ge "$need" ] ||
    die 16 "the local disk cannot hold the checkpoints: need ${need:-?} GB, free ${free:-?} GB"
  (
    while [ "${IB_PROBE:-0}" = 1 ] && [ "$NNODES" -gt 1 ] && [ ! -f "$OUT/ib-probe.json" ] && [ "$SECONDS" -lt 3600 ]; do
      sleep 15
    done
    began=$(date +%s)
    HF_XET_HIGH_PERFORMANCE=1 python3 "$SRC/stage_weights.py" --repo skt/A.X-K2-NVFP4 --revision "${NVFP4_IN#hf:}" \
      --out "$PWD/hub-model-nvfp4" > "$OUT/nvfp4-download.log" 2>&1 &&
      echo $(( $(date +%s) - began )) > /tmp/axk2-nvfp4.ready || touch /tmp/axk2-nvfp4.failed
  ) &
fi

bench_model() {  # <fp8|nvfp4>: the local checkpoint directory (waits for node0's NVFP4 download)
  case "$1:${FULL_REPO:-skt/A.X-K2}" in
    fp8:skt/A.X-K2|nvfp4:skt/A.X-K2-NVFP4) echo "$FULL_NATIVE"; return 0 ;;
  esac
  [ "$1" = nvfp4 ] && [ -n "${NVFP4_IN:-}" ] || return 1
  local waited=0
  until [ -f /tmp/axk2-nvfp4.ready ]; do
    { [ -f /tmp/axk2-nvfp4.failed ] || [ "$waited" -gt "${DOWNLOAD_TIMEOUT:-5400}" ]; } && return 1
    sleep 20; waited=$((waited + 20))
  done
  echo "$PWD/hub-model-nvfp4"
}

bench_alive() {  # a live (not zombie) process in the session of $1
  if command -v ps > /dev/null; then ps -o stat= -s "$1" 2>/dev/null | grep -qv '^Z'; else kill -0 -- "-$1" 2>/dev/null; fi
}

bench_stop() {  # stops the whole session of SERVER_PID: API servers, engine cores, local workers
  [ -n "$SERVER_PID" ] || return 0
  kill -INT -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 45); do bench_alive "$SERVER_PID" || break; sleep 2; done
  kill -9 -- "-$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  sleep "${BENCH_STOP_SETTLE:-10}"
}

# bench_clean <tag> <before|after> <nodes>: no GPU compute process and under 2 GiB used on every GPU of the
# first <nodes> nodes, no Ray placement group or GPU reservation left, port 8000 free. Waits up to 180 s.
bench_clean() {
  python3 - "$OUT/clean-$1-$2.json" "$3" "$NODE_IP" <<'EOF'
import json, socket, sys, time
path, nodes, head = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def gpus():
    import subprocess
    query = lambda q: subprocess.run(["nvidia-smi", q, "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True).stdout.split("\n")
    used = [int(float(v)) for v in query("--query-gpu=memory.used") if v.strip()]
    return {"compute_apps": sum(1 for a in query("--query-compute-apps=pid,used_memory") if a.strip()),
            "max_memory_used_mib": max(used or [0])}

def state():
    with socket.socket() as probe:
        probe.settimeout(2)
        result = {"port_8000_free": probe.connect_ex(("127.0.0.1", 8000)) != 0, "nodes": {"node0": gpus()}}
    try:
        import ray
        if not ray.is_initialized():
            ray.init(address="auto", logging_level="ERROR")
        ips = sorted({n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]} - {head})
        remote = ray.remote(num_cpus=0)(gpus)
        for index, ip in enumerate(ips[:nodes - 1], start=1):
            result["nodes"][f"node{index}"] = ray.get(remote.options(resources={f"node:{ip}": 0.001}).remote(), timeout=120)
        result["placement_groups"] = sum(pg.get("state") != "REMOVED" for pg in ray.util.placement_group_table().values())
        result["ray_gpus_free"] = ray.available_resources().get("GPU", 0) >= ray.cluster_resources().get("GPU", 0)
    except Exception as exc:
        result["ray_error"] = str(exc)[:300]
    return result

def clean(r):
    return (r["port_8000_free"] and not r.get("placement_groups") and r.get("ray_gpus_free", True)
            and all(n["compute_apps"] == 0 and n["max_memory_used_mib"] < 2048 for n in r["nodes"].values()))

deadline = time.time() + 180
while True:
    result = state()
    if clean(result) or time.time() > deadline:
        break
    time.sleep(15)
result["clean"] = clean(result)
json.dump(result, open(path, "w"))
print(json.dumps(result))
sys.exit(0 if result["clean"] else 1)
EOF
}

# bench_effective <tag> <expected max_model_len, -1 for auto-fit>: the engine's full config line and the lines
# that show what vLLM actually chose; exits 1 when the server contradicts the pinned settings.
bench_effective() {
  python3 - "$OUT/vllm-$1.log" "$2" "$NODE_IP" "$OUT/ray-nodes.json" <<'EOF'
import json, os, re, sys
log, expected, head, nodes_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
ips = [head] + (sorted({n["ip"] for n in json.load(open(nodes_file))} - {head}) if os.path.exists(nodes_file) else [])
patterns = {"async_scheduling": r"Asynchronous scheduling is (enabled|disabled)",
            "api_server_count": r"api_server_count|ApiServer_\d", "ray": r"Using ray distributed inference because",
            "uneven_pp": r"Hidden layers were unevenly partitioned", "dp_placement": r"Creating placement groups for data parallel",
            "executor": r"RayWorkerProc|RayWorkerWrapper|RayExecutorV2|MultiprocExecutor|WorkerProc",
            "cudagraph": r"cudagraph_mode|CUDAGraphMode", "dp_nccl": r"Disabling NCCL for DP synchronization",
            "dcp": r"[Dd]ecode context parallel|\bDCP\b", "expert_placement": r"Expert placement strategy",
            "kv_cache": r"GPU KV cache size|Maximum concurrency for", "loading": r"Loading weights took|Model loading took",
            "errors": r"\bERROR\b|Traceback"}
lines, engine, capture = {k: [] for k in patterns}, "", {}
for line in open(log, errors="replace"):
    if "NCCL INFO" in line:
        continue
    if not engine and "Initializing a V1 LLM engine" in line:
        engine = line.split("with config: ", 1)[-1].strip()
    for key, pattern in patterns.items():
        if re.search(pattern, line) and len(lines[key]) < 8:
            lines[key].append(re.sub(r"^.*?\[[\w./-]+:\d+\] ", "", line.strip())[:400])
    if "Graph capturing finished" in line:
        found = re.search(r"\bip=([\d.]+)", line)
        node = f"node{ips.index(found.group(1))}" if found and found.group(1) in ips else "node0"
        capture[node] = capture.get(node, 0) + 1
found = re.search(r"max_seq_len=(\d+)", engine)
result = {"max_model_len": int(found.group(1)) if found else None, "enforce_eager": "enforce_eager=True" in engine,
          "graph_capture_by_node": capture, "lines": lines, "problems": []}
if expected > 0 and result["max_model_len"] != expected:
    result["problems"].append(f"max_model_len {result['max_model_len']} instead of {expected}")
if result["enforce_eager"] or re.search(r"CUDAGraphMode\.NONE", engine):
    result["problems"].append("CUDA graphs are off")
if not capture:
    result["problems"].append("no CUDA graph capture in the log")
if any("disabled" in line for line in lines["async_scheduling"]):
    result["problems"].append("asynchronous scheduling is disabled although the protocol pins it on")
if not engine:
    result["problems"].append("no engine config line")
result["engine_config"] = engine
text = json.dumps(result)
text = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", re.sub(r"/mnt/azureml/cr/j/[^/\"']+", "<job>", text))
print(text)
sys.exit(1 if result["problems"] else 0)
EOF
}

# bench_nccl <since epoch>: transports of the NCCL processes started since then, on every node (counts only;
# addresses never leave the node). Exits 1 if any channel or communicator used NET/Socket or none used IB.
bench_nccl() {
  python3 - "$BENCH_NCCL_DIR" "$1" "$NODE_IP" <<'EOF'
import json, sys
root, since, head = sys.argv[1], float(sys.argv[2]), sys.argv[3]

def scan(root, since):
    import glob, os, re
    out = {"files": 0, "channels": {}, "networks": {}, "communicators": {}}
    for path in glob.glob(os.path.join(root, "*.log")):
        if os.path.getmtime(path) < since:
            continue
        out["files"] += 1
        for line in open(path, errors="replace"):
            found = re.search(r" via (NET/Socket|NET/IB|P2P|SHM|NET/\w+|COLLNET)", line)
            if found:
                out["channels"][found.group(1)] = out["channels"].get(found.group(1), 0) + 1
            found = re.search(r"Using network (\w+)", line)
            if found:
                out["networks"][found.group(1)] = out["networks"].get(found.group(1), 0) + 1
            found = re.search(r"rank (\d+) nranks (\d+) .*?commId (0x[0-9a-f]+) - Init COMPLETE", line)
            if found:
                comm = out["communicators"].setdefault(found.group(3), {"nranks": int(found.group(2)), "ranks_here": 0})
                comm["ranks_here"] += 1
    return out

result = {"node0": scan(root, since)}
try:
    import ray
    ray.init(address="auto", logging_level="ERROR")
    remote = ray.remote(num_cpus=0)(scan)
    for index, ip in enumerate(sorted({n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]} - {head}), start=1):
        result[f"node{index}"] = ray.get(remote.options(resources={f"node:{ip}": 0.001}).remote(root, since), timeout=300)
except Exception as exc:
    result["ray_error"] = str(exc)[:300]
nodes = [v for k, v in result.items() if k.startswith("node")]
socket = sum(n["channels"].get("NET/Socket", 0) + n["networks"].get("Socket", 0) for n in nodes)
ib = sum(n["channels"].get("NET/IB", 0) + n["networks"].get("IB", 0) for n in nodes)
result["gate"] = {"socket": socket, "ib": ib, "nodes_with_logs": sum(n["files"] > 0 for n in nodes),
                  "passed": socket == 0 and ib > 0 and len(nodes) > 1 and all(n["files"] for n in nodes)}
print(json.dumps(result))
sys.exit(0 if result["gate"]["passed"] else 1)
EOF
}

# bench_replica start <tag> <json args> <json env> | stop: dx2-D's second D server on node1, owned by a detached
# Ray actor (own session; the whole group is stopped). `start` prints node1's address.
bench_replica() {
  python3 - "$PWD/outputs" "$NODE_IP" "$@" <<'EOF'
import json, sys
import ray
root, head, action = sys.argv[1], sys.argv[2], sys.argv[3]
ray.init(address="auto", logging_level="ERROR", namespace="axk2-bench")

@ray.remote(num_cpus=0)
class Replica:
    def start(self, tag, args, env, root):
        import glob, os, subprocess
        out = next(d for d in sorted(glob.glob(os.path.join(root, "node*"))) if not d.endswith("node0"))
        model = open("/tmp/axk2-full.dir").read().strip()
        count = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.split("\n")) - 1
        env = {**os.environ, **env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(count)))}
        self.proc = subprocess.Popen(["vllm", "serve", model, *args], stdout=open(os.path.join(out, f"vllm-{tag}.log"), "w"),
                                     stderr=subprocess.STDOUT, env=env, start_new_session=True)
        return self.proc.pid

    def stop(self):
        import os, signal, subprocess
        for sig, wait in ((signal.SIGINT, 90), (signal.SIGKILL, 30)):
            try:
                os.killpg(self.proc.pid, sig)
                self.proc.wait(wait)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return self.proc.poll()

if action == "start":
    tag, args, env = sys.argv[4], json.loads(sys.argv[5]), json.loads(sys.argv[6])
    ip = sorted({n["NodeManagerAddress"] for n in ray.nodes() if n["Alive"]} - {head})[0]
    actor = Replica.options(name="replica", lifetime="detached", get_if_exists=True,
                            resources={f"node:{ip}": 0.001}).remote()
    ray.get(actor.start.remote(tag, args, env, root), timeout=300)
    print(ip)
else:
    actor = ray.get_actor("replica")
    print(json.dumps({"returncode": ray.get(actor.stop.remote(), timeout=300)}))
    ray.kill(actor)
EOF
}

bench_url_ok() { python3 -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=10)" "$1" 2>/dev/null; }

bench_wait_url() {  # <url> <seconds>
  local waited=0
  until bench_url_ok "$1"; do
    [ "$waited" -gt "$2" ] && return 1
    sleep 10; waited=$((waited + 10))
  done
}

# protocol_phase <name>: one protocol phase: settings from slo_suites.py phase-spec, a strict start, the
# effective-config and NCCL gates, the client suites of the phase, then a full stop and a clean-state record.
protocol_phase() {
  local name=$1 began spec model len rc commit base=http://127.0.0.1:8000 peer="" lb="" note="" rates_var
  local SPEC_LABEL SPEC_LAYOUT SPEC_RATES_KEY SPEC_CHECKPOINT SPEC_NODES SPEC_REPLICAS SPEC_LEN SPEC_SEQS SPEC_MEM
  local SPEC_BATCHED SPEC_SUITES SPEC_OPTIONS
  local -a SPEC_ARGS=() SPEC_ENV=()
  began=$(date +%s)
  spec=$(python3 "$BENCH/slo_suites.py" phase-spec "$name" --round "$ROUND") ||
    { phase_result "$name" failed "$began" "not a phase of slo_protocol.json"; return 1; }
  eval "$spec"
  model=$(bench_model "$SPEC_CHECKPOINT") ||
    { phase_result "$name" failed "$began" "the $SPEC_CHECKPOINT checkpoint is not staged on node0"; return 1; }
  if [ "$SPEC_CHECKPOINT" = fp8 ] && [ "$SPEC_NODES" -gt 1 ] && [ "$IB_MODE" != ib ]; then
    phase_result "$name" failed "$began" "the IB probe did not choose InfiniBand; an FP8 two-node phase never runs over TCP"
    return 1
  fi
  if ! bench_clean "$name" before $((SPEC_NODES * SPEC_REPLICAS)) > /dev/null; then
    report "node0.$name.clean_before" --file "$OUT/clean-$name-before.json"
    phase_result "$name" failed "$began" "a node was not clean before the start (GPU processes, port 8000 or Ray placement groups)"
    return 1
  fi
  nvidia-smi nvlink -gt d > "$OUT/nvlink-$name-before.txt" 2>&1
  log "vllm serve [$name] $model ${SPEC_ARGS[*]} (${SPEC_ENV[*]})"
  env "${SPEC_ENV[@]}" VLLM_ENGINE_READY_TIMEOUT_S="$FULL_HEALTH_TIMEOUT" \
    setsid vllm serve "$model" "${SPEC_ARGS[@]}" > "$OUT/vllm-$name.log" 2>&1 &
  SERVER_PID=$!
  if [ "$SPEC_REPLICAS" -gt 1 ]; then
    peer=$(bench_replica start "$name-replica1" "$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${SPEC_ARGS[@]}")" \
           "$(python3 -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:])))' "${SPEC_ENV[@]}" \
              "VLLM_ENGINE_READY_TIMEOUT_S=$FULL_HEALTH_TIMEOUT")" | tail -n 1)
    [[ "$peer" =~ ^[0-9.]+$ ]] || { peer=""; note="the second replica could not be started on node1"; }
  fi
  if [ -n "$note" ]; then
    :
  elif ! wait_healthy "$name" "$FULL_HEALTH_TIMEOUT"; then
    note="the server never became healthy (no --enforce-eager retry in a comparison phase)"
  elif [ -n "$peer" ] && ! bench_wait_url "http://$peer:8000/health" "$FULL_HEALTH_TIMEOUT"; then
    note="the second replica never became healthy"
  elif ! bench_effective "$name" "$SPEC_LEN" > "$OUT/effective-$name.json"; then
    note="the effective configuration contradicts the protocol: $(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['problems']))" "$OUT/effective-$name.json")"
  elif [ "$SPEC_CHECKPOINT" = fp8 ] && [ "$SPEC_NODES" -gt 1 ] && ! bench_nccl "$began" > "$OUT/nccl-$name.json"; then
    note="an NCCL communicator used NET/Socket or no InfiniBand transport was logged"
  fi
  report "node0.$name.facts" --text "$(server_facts "$name" | tail -c 7600)"
  report "node0.$name.parallel" --text "$(parallel_facts "$name")"
  [ -f "$OUT/effective-$name.json" ] && report "node0.$name.effective" --file "$OUT/effective-$name.json"
  [ -f "$OUT/nccl-$name.json" ] && report "node0.$name.nccl" --file "$OUT/nccl-$name.json"
  if [ -z "$note" ]; then
    len=$SPEC_LEN
    [ "$len" = -1 ] && len=$(python3 -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=30))['data'][0]['max_model_len'])" 2>/dev/null)
    report "node0.$name.startup" --text "{\"seconds_to_healthy\": $(cat "$OUT/startup-$name.seconds"), \"nccl_ib\": \"$IB_MODE\", \"max_model_len\": ${len:-null}}"
    if [ -n "$peer" ]; then
      python3 "$BENCH/lb_proxy.py" --listen 127.0.0.1:8100 --backend http://127.0.0.1:8000 \
        --backend "http://$peer:8000" > "$OUT/lb-$name.log" 2>&1 &
      lb=$!
      base=http://127.0.0.1:8100
      bench_wait_url "$base/health" 120 || note="the load balancer never became healthy"
    fi
  else
    report "node0.$name.failure" --text "$(failure_tail "$name")"
  fi
  if [ -z "$note" ]; then
    rates_var="SLO_RATES_$SPEC_RATES_KEY"
    commit=${AXK2_SRC_GIT:-unknown}
    AXK2_REPORT_PKGS="$PKGS" python3 "$BENCH/slo_suites.py" run --base "$base" --tag "$name" --suites "$SPEC_SUITES" \
      --tokenizer "$model" --max-model-len "${len:-$SPEC_LEN}" --max-num-batched-tokens "$SPEC_BATCHED" \
      --out-dir "$OUT/verify" --round "$ROUND" --layout "$SPEC_LABEL" --phase "$name" --rates-key "$SPEC_RATES_KEY" \
      --options "$SPEC_OPTIONS" --rates "${!rates_var:-${SLO_RATES:-}}" --soak-rates "${SOAK_RATES:-}" \
      --realtext-path "${REALTEXT_JSONL:-}" --commit "${commit##*@}" > "$OUT/client-$name.log" 2>&1
    rc=$?
    [ -n "$lb" ] && python3 -c "import urllib.request; print(urllib.request.urlopen('$base/lb/stats', timeout=10).read().decode())" \
      > "$OUT/lb-stats-$name.json" 2>/dev/null
    if [ "$rc" = 0 ]; then
      phase_result "$name" passed "$began"
    else
      report "node0.$name.client_failure" --text "$(tail -c 2500 "$OUT/client-$name.log"; failure_tail "$name" | tail -c 5000)"
      note="client failed"
    fi
  fi
  [ -n "$note" ] && phase_result "$name" failed "$began" "$note"
  [ -n "$lb" ] && kill "$lb" 2>/dev/null
  [ "$SPEC_REPLICAS" -gt 1 ] && bench_replica stop > "$OUT/replica-$name-stop.json" 2>&1
  bench_stop
  nvidia-smi nvlink -gt d > "$OUT/nvlink-$name-after.txt" 2>&1
  bench_clean "$name" after $((SPEC_NODES * SPEC_REPLICAS)) > /dev/null ||
    log "[$name] not clean after the stop; the next phase checks again before it starts"
  report "node0.$name.clean_after" --file "$OUT/clean-$name-after.json"
  [ -z "$note" ]
}
