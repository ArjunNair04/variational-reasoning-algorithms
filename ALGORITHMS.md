# Algorithms

## Finite-support EM

For a question $x$, answer $y^\star$, and latent rationale $h$, the model defines

$$
p_\theta(y^\star\mid x)
= \sum_h p_\theta(h\mid x)
  p_\theta(y^\star,\mathrm{EOS}\mid x,h).
$$

The latent sum is generally too large to compute directly, so we approximate it with a small support of sampled rationales. If $w_i$ is the detached responsibility of trace $h_i$, the joint M-step is

$$
\mathcal L(\theta)
= -\frac{1}{|M|}\sum_{x\in M}\sum_i w_i
\left[
  \log p_\theta(h_i\mid x)
  + \log p_\theta(y^\star,\mathrm{EOS}\mid x,h_i)
\right].
$$

The terminal answer marker is part of the rationale sequence. The numerical answer and one tokenizer EOS token form the answer target.

### PIS

PIS draws a fresh multiset from the current question-only policy. The proposal density therefore cancels from the self-normalised importance ratio:

$$
w_i
= \operatorname{softmax}_i
  \log p_{\theta_k}(y^\star,\mathrm{EOS}\mid x,h_i).
$$

The setting used here has eight traces for each of eight questions and holds the weights fixed for four local updates.

### Q5

Q5 proposes traces with an answer-derived prompt, reconstructs them under the ordinary question-only prompt, and keeps a token-unique FIFO support of size 16. Its posterior is

$$
w_i
= \operatorname{softmax}_i
\left\{
  \log p_{\theta_k}(h_i\mid x)
  + \log p_{\theta_k}(y^\star,\mathrm{EOS}\mid x,h_i)
\right\}.
$$

The setting used here has one local update. Q5-MORE changes only the number of raw proposals from 16 to 32 before compression to the same support size. The available Q5-MORE result is preliminary.

The Q5 implementation uses this uncorrected posterior with answer-derived proposals.

#### Large-support posterior-sampled M-step

The buffer-sampling follow-up separates the support retained for posterior
estimation from the traces used by one M-step.  It retains up to 32 unique Q5
traces for each selected question.  The full-support cell applies the ordinary
weighted objective to all retained traces.  The sampled cell instead draws 16
indices independently from the detached posterior,

$$
i_j \sim \operatorname{Categorical}(w_1,\ldots,w_K),
\qquad j=1,\ldots,16,
$$

and applies the M-step to their empirical multiplicities.  Conditional on the
finite support, this is an unbiased Monte Carlo estimate of the same weighted
objective.  Duplicate draws are aggregated before the forward and backward
passes, while the next responsibility calculation still uses the complete
retained support.  A sample size of zero is the behavior-locked full-support
path.

Because the fixed question schedule visits each optimisation question once,
this study changes within-question support depth rather than testing replay of
the same question across outer rounds.

#### Support-allocation follow-up

The follow-up tests two ways to spend more computation without changing Q5's
posterior. The first draws 64 candidates per selected question, keeps 32 unique
FIFO entries under the existing buffer rule, and applies the exact full-support
M-step. The question schedule remains four questions per round, so this changes
within-question search depth without changing question breadth.

The second retains the 32-candidate control and evaluates the largest posterior
term exactly. If $t=\arg\max_i w_i$ and $R=\sum_{i\ne t}w_i$, it draws 15
indices from the normalized residual posterior,

$$
j_m\sim\operatorname{Categorical}\left(\frac{w_i}{R}:i\ne t\right),
\qquad m=1,\ldots,15,
$$

and estimates the weighted M-step gradient by

$$
\widehat g
=w_tg_t+\frac{R}{15}\sum_{m=1}^{15}g_{j_m}.
$$

Conditional on the retained finite support, this estimator is unbiased for
$\sum_iw_ig_i$. It cannot waste repeated draws on the dominant trace, while
ordinary posterior-categorical sampling can do so when responsibilities are
highly concentrated. Sampling uses an isolated deterministic random stream;
proposal generation, question order and the next E-step remain unchanged.

### Related EM presets

VIN and VOUT use the same joint posterior with persistent question-only support. VIN refreshes responsibilities at each local update; VOUT freezes them for the outer round. POLD uses fresh question-only support with uniform empirical weights. These are settings of the same update, not separate implementations.

### Answer-conditioned importance correction

The AC-PIS diagnostic samples from an answer-conditioned proposal $g(h\mid x,y^\star)$ and applies the exact self-normalised correction

$$
w_i=\operatorname{softmax}_i
\left[
\log p_{\theta_k}(h_i\mid x)
+\log p_{\theta_k}(y^\star,\mathrm{EOS}\mid x,h_i)
-\log g(h_i\mid x,y^\star)
\right].
$$

The M-step is unchanged.

### Centred trace credit

The centred diagnostic keeps ordinary positive answer training but gives the rationale a signed relative coefficient:

$$
c_i^{h}=w_i-\frac{1}{K},
\qquad
c_i^{y}=w_i.
$$

The rationale coefficients sum to zero. The experiments examined how these negative coefficients affected training when the sampled support was weak.

### Null-state abstention

The smooth abstention diagnostic adds one fixed null state. For $K$ real traces,

$$
q_i=
\frac{(1-\pi_0)K^{-1}\exp(\ell_i/\tau)}{Z},
\qquad
q_0=
\frac{\pi_0\exp(b_0/\tau)}{Z}.
$$

The real M-step coefficients are the unconditional $q_i$, so $\sum_i q_i=1-q_0$. Renormalising them back to one would remove the abstention mechanism.

## GRPO

For the responses to one question, GRPO standardises the verifier reward:

$$
A_i = \frac{r_i-\bar r}{\operatorname{sd}(r)+10^{-8}}.
$$

If the group has no reward variation, every advantage is zero. The implementation here uses the clipped token-level ratio objective

$$
\min(\rho_{it}A_i,
      \operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i)
-\beta\,K_3,
$$

where $\rho_{it}=\exp(\ell_{it}-\ell^{\mathrm{old}}_{it})$ and

$$
K_3 = \exp(\ell^{\mathrm{ref}}_{it}-\ell_{it})
      -(\ell^{\mathrm{ref}}_{it}-\ell_{it})-1.
$$

The loss is averaged over active response tokens.

## RLOO

RLOO first forms the sampled KL-shaped return

$$
R_i=r_i-\beta
(\log p_\theta(o_i\mid x)-\log p_{\mathrm{ref}}(o_i\mid x)),
$$

then subtracts the mean return of the other responses to the same question:

$$
A_i=R_i-\frac{1}{G-1}\sum_{j\ne i}R_j.
$$

The policy loss is the negative mean of $A_i\log p_\theta(o_i\mid x)$.

## TRICE

TRICE keeps one persistent trace for each question. The configuration here initialises each chain once with an answer-derived prompt, then uses question-only proposals. A correct proposal replaces the chain state; an incorrect proposal is rejected. For valid retained states, the control-variate estimator has terms

$$
\frac{1}{\sum_m c'_m}\sum_m c'_m
\left[
\nabla\log p_\theta(z'_m\mid x_m)
-\beta_m\nabla\log p_\theta(\widetilde z_m\mid x_m)
\right],
$$

with the leave-one-out scale

$$
\beta_m=
\frac{\sum_{j\ne m}c'_j\widetilde c_j}
     {\sum_{j\ne m}c'_j}.
$$

The function returns detached coefficients and roles. A training loop can then evaluate the corresponding trace log probabilities.

## Supervised and self-training baselines

All four baselines use ordinary maximum-likelihood training on a selected completion and an explicit EOS target.

- **Gold-CoT SFT** trains directly on the supplied human rationale.
- **RFT** generates once, retains answer-correct naturally terminated traces, then trains on them.
- **ReST-EM** repeats Generate and Improve phases, resetting to the original adapter before each Improve phase.
- **STaR** first tries one greedy rationale. If it fails, it generates a rationale with the known answer as a hint, removes the hint from the training context, and trains the retained completion under the ordinary question prompt.

The self-training module selects traces that can be passed to a token-level maximum-likelihood training loop.
