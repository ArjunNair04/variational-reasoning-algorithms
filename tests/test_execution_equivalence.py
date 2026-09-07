"""CPU comparisons with functions read from the archived execution source.

Only selected original definitions are compiled; importing this module never
loads a language model, tokenizer, dataset, or the experiment driver.
"""
import ast
from contextlib import nullcontext
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from variational_reasoning.em import (
    joint_weights, pis_weights, importance_weights, uniform_weights,
    weighted_joint_loss, UniqueFIFOSupport,
)
from variational_reasoning.diagnostics import effective_sample_size, entropy, temperature_weights
from variational_reasoning.answer_events import parse_gsm8k_answer_event
from variational_reasoning.answer_targets import terminated_answer_ids

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'experiments' / json.loads((ROOT / 'experiments/studies.json').read_text())['selected_methods']['source_bundle']


def source(path):
    with tarfile.open(ARCHIVE) as archive:
        return archive.extractfile(path).read()


def original(names, **environment):
    """Read the actual function AST, including all original branches."""
    tree = ast.parse(source('lm_study/ac_alg1.py'))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign, ast.AnnAssign))
                and (getattr(node, 'name', None) in names or
                     isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    env = {'np': np, 'math': math, 'nullcontext': nullcontext, **environment}
    exec(compile(module, 'original/ac_alg1.py', 'exec'), env)
    return env


def test_copied_parsers_are_byte_identical_to_executed_source():
    for name in ['answer_events.py', 'answer_targets.py']:
        assert (ROOT / 'src/variational_reasoning' / name).read_bytes() == source('lm_study/' + name)


@pytest.mark.parametrize('text,answer,strict', [
    ('12 + 3 = 15\n#### 15', 15, True),
    ('The answer is 15.', 15, False),
    ('#### 15\n#### 15', 15, False),
    ('#### 15 pounds', 15, False),
    ('#### +15', 15, True),
    ('#### 1,500', 1500, True),
])
def test_extracted_answer_and_terminal_format_are_separate(text, answer, strict):
    legacy = parse_gsm8k_answer_event(text, mode='legacy')
    terminal = parse_gsm8k_answer_event(text, mode='strict_terminal_marker')
    assert legacy.answer == answer
    assert terminal.strict_valid is strict
    # No EOS argument: natural termination is recorded independently.


def test_eos_added_once_and_historical_no_eos_preserved():
    tok = SimpleNamespace(eos_token_id=99)
    assert terminated_answer_ids(tok, [15], termination='eos') == [15, 99]
    assert terminated_answer_ids(tok, [15, 99], termination='eos') == [15, 99]
    assert terminated_answer_ids(tok, [15], termination='none') == [15]


def test_actual_prefix_admission_preserves_whole_tokens_and_excludes_answer():
    fn = original({'_prefix_ids_through_marker'})['_prefix_ids_through_marker']
    class Tokenizer:
        pieces = {1: 'work\n', 2: '#### ', 3: '15', 4: '#### 15', 5: '##', 6: '##\t'}
        def decode(self, ids): return ''.join(self.pieces[i] for i in ids)
    tok = Tokenizer()
    for ids, expected in [([1, 2, 3], [1, 2]), ([1, 4], []), ([1, 5, 6, 3], [1, 5, 6])]:
        text = tok.decode(ids)
        kept, valid = fn(tok, ids, text, text.index('####') + 4)
        assert kept == expected
        assert valid is bool(expected)


def test_actual_delta_proposal_and_target_prompt_are_different():
    env = original({'_build_proposal_prompt', 'PROPOSAL_PROMPTS', 'TRACE_REPRESENTATIONS'})
    prompt = 'Question: example\nAnswer: working\n#### 9\n\nQuestion: balance\nAnswer:'
    fn = env['_build_proposal_prompt']
    assert fn(prompt, 15, 'question') == prompt
    proposed = fn(prompt, 15, 'answer_derive')
    assert proposed == prompt[:-len('\nAnswer:')] + '\nThe correct final answer is 15. Derive a step-by-step solution and finish with #### 15.\nAnswer:'
    assert prompt.endswith('Question: balance\nAnswer:')  # original is not modified


def test_fifo_matches_original_for_unlabelled_buffer():
    fn = original({'_enforce_buffer_limit'})['_enforce_buffer_limit']
    rows = []; clean = UniqueFIFOSupport(2)
    for ids in [(1,), (2,), (1,), (3,), (4,)]:
        if not any(row.ids == ids for row in rows):
            rows.append(SimpleNamespace(ids=ids, is_gold=False))
        fn(rows, 2, 'fifo')
        clean.add(ids)
        assert [row.ids for row in rows] == clean.items


def test_original_estimator_logits_and_detached_weights_match_clean_kernels():
    torch = pytest.importorskip('torch')
    env = original({'_barber_variational_logits', 'VARIATIONAL_ESTIMATORS',
                    '_posterior_weights', '_one_sided_ess_temperature',
                    '_responsibility_effective_sample_size', 'RESPONSIBILITY_POSTERIORS'},
                   torch=torch, MULTI_VERIFIER_POSTERIORS=())
    rng = np.random.default_rng(8724)
    for _ in range(12):
        trace, answer, proposal = -rng.exponential(3, (3, 7))
        tensors = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (trace, answer, proposal)]
        for name, clean in [
            ('delta_joint', joint_weights(trace, answer, [0]*7)),
            ('prior_importance', pis_weights(answer, [0]*7)),
            ('uniform_mc', uniform_weights([0]*7)),
            ('answer_conditioned_importance', importance_weights(trace, answer, proposal, [0]*7)),
        ]:
            logits = env['_barber_variational_logits'](name, *tensors[:2], proposal_trace_logprobs=tensors[2])
            weights, _ = env['_posterior_weights'](logits, 'softmax_entropy', temperature=1, ess_floor_fraction=0)
            assert not weights.requires_grad
            np.testing.assert_allclose(weights.numpy(), clean, atol=1e-12)


def test_temperature_floor_matches_original_and_preserves_hard_exclusions():
    torch = pytest.importorskip('torch')
    env = original({'_one_sided_ess_temperature', '_responsibility_effective_sample_size'}, torch=torch)
    for logits in [[0, 0, 0, 0], [0, -8, -12, -30], [0, -10, -np.inf, -2]]:
        weights, temperature = temperature_weights(logits, ess_floor_fraction=.5)
        original_temperature = env['_one_sided_ess_temperature'](torch.tensor(logits, dtype=torch.float64), 1, .5)
        assert temperature == pytest.approx(original_temperature, rel=2e-6)
        assert effective_sample_size(weights) >= .5 * np.isfinite(logits).sum() - 1e-10
        assert weights[np.isneginf(logits)].sum() == 0
        assert math.exp(entropy(weights)) + 1e-10 >= effective_sample_size(weights)


def test_original_objective_uses_sequence_sums_and_usable_question_mean():
    torch = pytest.importorskip('torch')
    # Exact original objective, with a differentiable table supplying token scores.
    def pad(tok, rows):
        ids = torch.tensor([r.ids for r in rows])
        span = ids >= 0
        ans = torch.zeros_like(span); ans[:, -1] = True
        return ids.clamp(min=0), span, ans
    def score(model, ids, mask, grad=True, length_norm=False):
        assert length_norm is False
        return (model[ids] * mask).sum(dim=1)
    env = original({'_B_unsup_for_questions', '_latent_mstep_components',
                    '_latent_mstep_mask', 'LATENT_MSTEP_OBJECTIVES'}, torch=torch,
                   DEV='cpu', _pad_trace_rows=pad, seq_logprobs=score)
    model = torch.tensor([-1., -2., -3., -4., -5., -6.], requires_grad=True)
    rows = {0: [SimpleNamespace(ids=[0, 1, 2]), SimpleNamespace(ids=[3, -1, 4])],
            1: [], 2: [SimpleNamespace(ids=[5, -1, -1])]}
    weights = {0: torch.tensor([.25, .75]), 2: torch.tensor([1.])}
    objective = env['_B_unsup_for_questions'](model, None, rows, [0, 1, 2], weights)
    clean = weighted_joint_loss([-3, -4, -6], [-3, -5, 0], [.25, .75, 1], [0, 0, 2])
    assert -objective.item() == pytest.approx(clean)
    (-objective).backward()
    np.testing.assert_allclose(model.grad.numpy(), [-.125, -.125, -.125, -.375, -.375, -.5])
    assert env['_B_unsup_for_questions'](model, None, {1: []}, [1], {}).item() == 0


def test_zero_mass_rows_cannot_poison_the_reference_objective():
    assert weighted_joint_loss([-2, -np.inf], [-1, -np.inf], [1, 0], [0, 0]) == 3
    np.testing.assert_allclose(joint_weights([-1, -np.inf], [-1, -2], [0, 0]), [1, 0])


def test_grpo_microbatch_accumulation_matches_original_active_token_objective():
    import textwrap
    torch = pytest.importorskip('torch')
    from variational_reasoning.policy_gradient import grpo_loss
    text = source('lm_study/grpo.py').decode()
    start = text.index('                ratio = torch.exp')
    stop = text.index('                with torch.no_grad()', start)
    code = textwrap.dedent(text[start:stop])
    current=np.array([[-1.,-2.,-4.],[-3.,-1.,-2.]])
    old=current+np.array([[.3,-.2,.1],[-.3,.4,-.4]])
    reference=current+.15;mask=np.array([[True,True,False],[True,True,True]])
    advantages=np.array([1.,-.8]);total=0
    for index in range(2):
        env={'torch':torch,'lp':torch.tensor(current[index:index+1]),'old_lp':torch.tensor(old),
             'ref_lp':torch.tensor(reference),'A':torch.tensor(advantages[:,None]),
             'm_tok':torch.tensor(mask,dtype=torch.float64),'sel':slice(index,index+1),
             'epoch_tokens':torch.tensor(mask.sum()),'optimizer_step_scope':'batch',
             'clip':.2,'kl_coef':.02}
        exec(compile(code,'original/grpo.py','exec'),env)
        total += env['loss'].item()
    close=grpo_loss(current,old,reference,advantages,mask,clip=.2,kl_coef=.02)
    assert total == pytest.approx(close, abs=1e-12)


def test_rloo_original_kl_shaping_and_leave_one_out():
    import textwrap
    torch=pytest.importorskip('torch')
    from variational_reasoning.policy_gradient import rloo_advantages
    text=source('lm_study/rloo.py').decode()
    start=text.index('        rew_kl =');stop=text.index('        adv_t =',start)
    rewards=np.array([1.,0.,1.,1.]);policy=np.array([-2.,-3.,-1.,-2.]);reference=policy-np.array([.4,.2,-.1,.3])
    env={'np':np,'rew':rewards,'lp_cur':torch.tensor(policy),'lp_ref':torch.tensor(reference),
         'kl_coef':.03,'pids':[0,1],'pid_row':np.array([0,0,1,1])}
    exec(compile(textwrap.dedent(text[start:stop]),'original/rloo.py','exec'),env)
    clean=rloo_advantages(rewards,[0,0,1,1],policy_logp=policy,reference_logp=reference,kl_coef=.03)
    np.testing.assert_allclose(env['adv'],clean)
    assert clean[2] != 0  # equal binary rewards do not eliminate KL-shaped advantage


def test_trice_original_leave_one_out_scales():
    from variational_reasoning.trice import Chain,Proposal,trice_step,control_variate_terms
    tree=ast.parse(source('lm_study/ac_alg1_trice.py'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='leave_one_out_acceptance_scales')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[]));env={}
    exec(compile(module,'original/ac_alg1_trice.py','exec'),env)
    transitions=[trice_step(Chain(i,'old',f'old{i}',True),Proposal(i,'new',f'new{i}',correct))[1]
                 for i,correct in enumerate([True,False,True])]
    adapted=[SimpleNamespace(after=SimpleNamespace(question_id=t.after.question_id,state_correct=t.after.correct),proposal=t.proposal) for t in transitions]
    _, clean=control_variate_terms(transitions)
    assert clean == env['leave_one_out_acceptance_scales'](adapted)


def test_original_split_reserve_is_fixed_and_disjoint_without_loading_data():
    tree=ast.parse(source('lm_study/tasks.py'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_gsm8k_train_validation_pools')
    env={'np':np,'_GSM8K_VALIDATION_SEED':20260716,'_GSM8K_VALIDATION_SIZE':400}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'original/tasks.py','exec'),env)
    train,validation=env['_gsm8k_train_validation_pools'](1000)
    assert len(validation)==400 and len(train)==600
    assert not set(train)&set(validation)
    assert set(train)|set(validation)==set(range(1000))
    train2,validation2=env['_gsm8k_train_validation_pools'](1000)
    np.testing.assert_array_equal(validation,validation2)
