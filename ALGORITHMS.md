# Update rules

Write the question as $q$, the complete known-answer suffix as $a$, and the
unobserved reasoning trace as $h$. In the final experiments, $h$ contains the
first `####` marker and $a$ contains the numerical answer followed by EOS. The
model therefore assigns the complete response probability

$$p_\theta(h,a\mid q)=p_\theta(h\mid q)p_\theta(a\mid q,h).$$

Professor Barber's finite-support formulation assigns nonnegative masses to
traces in a retained set $\mathcal B(q)$. Its ordinary E-step is

$$w_s = \frac{p_{\theta_{\mathrm{old}}}(h^s,a\mid q)}
 {\sum_{h\in\mathcal B(q)}p_{\theta_{\mathrm{old}}}(h,a\mid q)}.$$

These are the exact posterior masses **conditional on the retained set**. The
proposal determines which traces can enter that set. With the pseudo-posterior
proposal, the prompt supplies the known answer; the retained trace is then
scored and fitted under the original question-only prompt. No proposal-density
correction appears in this discrete-set E-step.

## Fitting the complete response

For $M$ usable questions, the detached-weight loss is

$$L(\theta)=-\frac1M\sum_q\sum_s w_s
 [\log p_\theta(h^s\mid q)+\log p_\theta(a\mid q,h^s)].$$

Log probabilities are sums over response tokens. They are not divided by trace
length. Empty questions are omitted from the mean; an entirely empty minibatch
has no latent update. Ordinary delta recomputes weights before each of $J$
optimizer steps. The fixed-weight diagnostic computes them once per minibatch;
these choices coincide at $J=1$.

The final delta buffer retains distinct original token prefixes and evicts the
oldest on overflow. A duplicate does not become a second atom or refresh its
age. Prefix admission requires an intact token boundary through the marker;
a token containing part of the generated numerical answer cannot be cut apart.
The sampled suffix is replaced with the known complete answer event.

## Importance sampling

Importance sampling retains fresh **occurrences**. If an eligible trace occurs
twice, both draws contribute. For proposal distribution $q(h)$,

$$\widetilde w_s=\frac{p_{\theta_{\mathrm{old}}}(h^s,a\mid q)}{q(h^s)},
\qquad w_s=\frac{\widetilde w_s}{\sum_j\widetilde w_j}.$$

For question-only prior draws the trace factor cancels, leaving the known
answer likelihood. Prefix eligibility conditions sampling on admission; its
common normalising probability also cancels among the retained draws. The
finite self-normalised estimate is not generally unbiased. With a
pseudo-posterior proposal, the recorded sample-time prefix probability remains
in the denominator. The final importance method holds its weights fixed for
all $J$ fitting steps. The historical correction pilot refreshed the numerator
while holding its proposal denominator fixed; it has its own archived version.

## Weight concentration

Classical effective sample size is $1/\sum_s w_s^2$, and entropy is
$H=-\sum_{s:w_s>0}w_s\log w_s$. Their effective counts obey
$\mathrm{ESS}\leq\exp(H)\leq K$. Here $K$ counts finite admissible scores.
Neither count measures calculation validity or independent optimiser updates.
The temperature intervention increases the softmax temperature only when the
normalised ESS falls below a requested floor. Negative-infinity exclusions stay
excluded. The code retains the original bounded search; a floor of one can
require an infinite temperature for exact equality.

## Comparison methods

GRPO centres and scales checker rewards within each question's response group.
The executed loss clips token probability ratios at $1\pm\epsilon$ and includes
a reference-policy KL penalty. Its numerical KL estimator clamps the reference
minus current token log ratio to $[-5,5]$ before exponentiation. The selected
configuration accumulates microbatches into one optimizer step per full batch.
An all-pass or all-fail group has zero centred task-reward advantage; the KL
term can still act.

RLOO first subtracts a sequence-level sampled KL penalty from each reward, then
subtracts the other responses' mean return. Its fixed advantages weight summed
sequence log probabilities across repeated epochs; the executed RLOO trainer
has no PPO clipping step. The reference singleton fallback matches the source,
although the thesis comparisons use groups larger than one.

TRICE retains a passing response for each question and tests a fresh proposal.
Its score estimator uses the retained state and a leave-one-out control-variate
coefficient on the proposal, including rejected proposals. The Qwen/LoRA
comparison initialises with the known answer available, then proposes from the
question-only prompt. This implementation choice is recorded separately from
the published algorithm's dataset-specific initialisation.

RFT, ReST-EM and STaR select checker-passing, naturally terminated responses for
maximum-likelihood fitting. ReST-EM repeats generation and improvement; STaR
also tries a known-answer rationale when its direct attempt fails. Gold SFT has
additional worked-solution supervision. Their exact resetting, grouping and
learning-rate choices are in the selected study YAML, rather than imposed by
the small selection helpers.

The comparator implementations credit their source papers in their module
docstrings: STaR (Zelikman et al.), ReST-EM (Singh et al.), TRICE (Phan et al.),
GRPO (Shao et al.) and RLOO (Ahmadian et al.). This repository's contribution is
the recorded language-model implementation and controlled comparisons, not the
invention of those methods. Historical centred-credit and null-state APIs remain
for compatibility and are not part of the principal thesis algorithms.
