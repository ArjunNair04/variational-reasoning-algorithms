# Learned posterior distribution: three-seed pilot

Run `2d7c8a61`, seeds 1201, 1213 and 1217. One new cell,
`Q5-POSTERIOR-U1`, compared with validated Q5 controls 7483302.1-3.
This implements the user's chosen **distribution replacement**, not merely a
learned proposal source for the old delta-mixture M-step.

## Question

Can a separately trained answer-conditioned generator turn Q5's concentrated
finite-buffer weights into a useful distribution over new reasoning traces?
The earlier fixed-weight continuation kernel lowered mean final accuracy, so
this pilot learns its sampling distribution instead of transferring weight to
an unscored alternative ending. It is not assumed to improve accuracy.

## Objects and update order

Write x for a question, a for its known answer, h for a rationale including the
answer marker, and a+ for the answer tokens followed by EOS. The main model has
parameters theta. A second LoRA adapter has parameters phi. They share only
the frozen Qwen backbone; each adapter has its own persistent Adam optimizer.
Phi starts as an exact copy of the initial theta adapter, not a copy refreshed
each round. Both use rank-16 attention-plus-MLP LoRA and learning rate 1e-5.

1. The **main** model generates 16 answer-derived proposals for each of four
   questions. The persistent unique FIFO-16 buffer and ordinary Q5 admission
   rules are unchanged. No gold rationale is introduced.
2. Compute the ordinary Q5 teacher weights on that buffer:

   `w_s = softmax_s[log p_theta(h_s | x) + log p_theta(a+ | x,h_s)]`.

   These are detached finite-buffer weights, not exact importance weights.
3. Make one posterior-adapter update, maximising the weighted sum of token log
   probabilities of each stored complete target `c_s = h_s ++ a+`, under the
   same answer-derived prompt A(x,a):

   `D(phi) = mean_x sum_s w_s log r_phi(c_s | A(x,a))`.

   The answer is supplied in this training-only prompt. The target includes
   the answer and EOS as well as the rationale; this is completion distillation,
   not an entropy-optimised E-step. Teacher scores and main parameters stay
   fixed during this update.
4. Draw 16 fresh completions independently from the updated r_phi per question,
   at temperature 1, top-k 0, top-p 1, with the existing 256-token cap. Do not
   deduplicate or reweight these draws. A deterministic map T extracts h.
5. Make one main-model update on those new traces under question-only prompts:

   `J(theta) = mean_x (1/16) sum_j log p_theta(T(c_j), a+ | x)`.

   Hold phi and the sampled targets fixed. Only this fresh support is used for
   the main M-step; the original buffer remains the teacher for later rounds.
   New posterior samples never enter it. Evaluation uses only theta and never
   reveals the answer.

The learned rationale law is the pushforward
`q_phi(h | x,a) = sum_{c:T(c)=h} r_phi(c | A(x,a))`, including the finite length
cap. The equal-weight Monte Carlo M-step estimates an expectation under this
law, replacing `sum_s w_s delta_hs`. Fitting r_phi to Q5 targets does not
guarantee improvement of the variational bound or recovery of the true posterior.
No proposal-density correction or extra entropy objective is introduced.

## Completion mapping

The first usable answer-marker boundary is kept in native sampled tokens.
When that boundary is absent, unalignable or has an empty rationale, take the
decoded text before the first marker, append a newline and `####`, and
retokenise this fallback only. Empty reasoning is permitted. Append the gold
answer and EOS to every mapped target. No rejection, resampling, correctness
filter or probability renormalisation occurs. Record every repair and the raw
completion, including its extracted correctness and strict formatting before
the gold target is attached. These answer-aware training diagnostics are not
held-out accuracy measurements.

## Fixed protocol and cost

Qwen3-1.7B-Base, three-shot prompts, 128 training questions, 32 rounds, four
questions per round, 16 teacher proposals per question, buffer size 16 and one
main update per round are unchanged. There is no additional KL or ESS rule.
The fixed 400-question training-derived validation pool is unchanged. The
official test split is not loaded. Main generations preserve their RNG stream;
adapter initialization, posterior fitting and posterior sampling use isolated
streams. Diagnostics do not draw random samples.

Per seed: 2,048 main teacher proposals plus 2,048 posterior completions; 32 main
updates plus 32 posterior updates. Report generation and backward-token costs
for both phases, as well as total GPU time. This is not an equal-compute
comparison. The earlier 5-15% overhead estimate excluded the new posterior
rollouts and must not be used as this study's ETA.

## Frozen analysis

Report Final extracted Acc@1 first, strict final second and normalized Acc@1
AUC over rounds 0-32 third. Reuse the receipt-bound controls only if source,
configuration, runtime, validation support and training-question schedule pass
the paired checks. Report all three seed differences and descriptive paired
95% t intervals. A positive mean final difference without mean strict-final
loss nominates a follow-up for discussion, not automatic submission.

Mechanism diagnostics: teacher concentration, posterior sample uniqueness,
exact teacher-support matches, raw proposal correctness/format, boundary
repairs, posterior gradient and parameter-change norms, phase timings and
main/posterior backward tokens. Require full round coverage, equal 1/16 sample
weights, no child insertion, and checksummed main and posterior adapters.
Three development seeds cannot establish generality or a publication-level win.

## Execution

The generator is `lm_study/generate_qwen3_17b_q5_learned_posterior.py`; edit it,
not the generated YAML. The submitter checks an offline tiny-Qwen two-adapter
smoke before creating a user-held three-task array. Publish immutable commit,
configuration hash and job IDs before release. A dependent validator executes
the prepared analyzer. There are no control reruns or automatic follow-ups.
