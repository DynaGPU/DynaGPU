# Best Practices Before Starting DynaGPU

_Written 2026-11-07. Roughly in the order they matter._

## 1. Prove the idea before building the project

- **Run week 1 of the spike (E1–E4) first.** Don't build Kubernetes code until weighted turns are shown to work on a real GPU.
- **Write pass/fail rules down before testing** (for example, the split must land within ±5% of target). That keeps the results honest when they're borderline.
- **Publish the results either way,** with GPU model, driver version and scripts. Reproducible numbers are what convince HAMi, NVIDIA and CNCF people.

## 2. Settle the name and legal basics

- **Pick the final name now.** Check domains (.io, .dev, .com), Docker Hub, the GitHub organization, and USPTO / WIPO trademarks. "DynaGPU" sits close to Dynamia (the company behind HAMi), Dynatrace's DynaKube and NVIDIA Dynamo.
- **License: Apache-2.0.** The CNCF default and easy for companies to adopt.
- **Use DCO sign-off (`git commit -s`)** instead of a contributor agreement, as CNCF projects do.
- **Check your employer's rules** on side projects, especially ones close to your work.
- **Be careful with driver code.** Use NVIDIA's open-kernel-module headers (MIT) only; never copy proprietary driver code. Credit Valve and RETARGET for methods; RETARGET has no license, so its code can't be reused.

## 3. Set up the repo properly from day one

- **Required files:** `README.md` (problem, the 70/30 example, status, what it is not), `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (CNCF's), `SECURITY.md`, `GOVERNANCE.md` (even "single maintainer for now").
- **State clearly what it is not:** not a security boundary, not for mutually untrusted tenants, compute only.
- **Protect `main`:** changes go through pull requests; CI builds, lints and runs tests.
- **Keep design records in `docs/design/`** so decisions (DRA-only, compute only, no per-launch interception) live in the repo.

## 4. Engineering habits for this project

- **Go** for the node agent and DRA parts; **C** for the small shim inside CUDA processes.
- **Isolate the driver calls behind one small interface,** with a per-driver-version table of control IDs. That is the biggest technical risk.
- **Fail open everywhere.** If the agent dies, pods fall back to NVIDIA's default equal turns. Test this from the start (E10).
- **Keep a test matrix** of GPU models × driver versions × CUDA versions.
- **Measure and publish DynaGPU's own overhead,** since "no per-launch cost" is a headline claim.

## 5. Community, and order of outreach

1. **HAMi maintainers first,** privately, after the week-1 results. Their DRA driver already publishes compute; pitch DynaGPU as complementary, not a rival.
2. **Kubernetes WG Device Management** (DRA), to propose a standard `compute` capacity.
3. **Public launch:** a blog post with the 70/30 before-and-after graph, then KubeCon talks and Slack.
4. **CNCF Sandbox application** once someone outside the core team uses it.

Keep the plan public: a roadmap issue, "good first issue" labels and a regular update cadence.

## 6. Protect your time

- **Stay narrow:** compute time-sharing only. Say no to memory, scheduling and multi-cluster features for now.
- **Expect that NVIDIA could ship weighted turns itself.** Staying vendor-neutral and upstream-friendly protects the project if it does.

## Top three

1. Run the week-1 spike before anything else.
2. Settle the name and license now.
3. Talk to the HAMi maintainers before announcing.
