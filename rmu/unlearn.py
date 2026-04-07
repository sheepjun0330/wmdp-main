import datetime
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import tqdm as tqdm
import wandb
from torch.optim import AdamW

from rmu.utils import forward_with_cache, get_data, get_params, load_model


SUPPORTED_DUAL_MODES = {
    "alm_sam_sam_joint2",
    "base_adamw_adamw_joint_alm",
    "base_adamw_alm_joint",
    "robust_unlearn_sam_alm",
    "sharp_minmax_nomask_alm",
}


@dataclass
class TrainingState:
    params: list
    adam_state_r: defaultdict
    adam_state_f: defaultdict
    lam: torch.Tensor
    frozen_module: object
    updated_module: object
    joint_optimizer: object = None


@dataclass
class StepContext:
    args: object
    updated_model: object
    frozen_model: object
    tokenizer: object
    state: TrainingState
    unlearn_batch: list
    retain_batch: list
    control_vec: torch.Tensor
    topic_idx: int
    max_length: int


def run_rmu(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    _print_rmu_config(args)
    _validate_dual_mode(args.dual_mode)

    updated_model = updated_model.train()
    params = _resolve_update_params(updated_model, args)
    state = _build_training_state(updated_model, frozen_model, args, params)
    control_vectors_list = _build_control_vectors(updated_model, forget_data_list, args)
    num_batches = _compute_num_batches(forget_data_list, retain_data_list, args.max_num_batches)

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    try:
        for epoch in range(args.epochs):
            print(f"======= Epoch {epoch} =======")
            with tqdm.tqdm(total=num_batches) as pbar:
                for idx in range(num_batches):
                    topic_idx = idx % len(forget_data_list)
                    batch_idx = idx // len(forget_data_list)
                    step_ctx = StepContext(
                        args=args,
                        updated_model=updated_model,
                        frozen_model=frozen_model,
                        tokenizer=tokenizer,
                        state=state,
                        unlearn_batch=forget_data_list[topic_idx][batch_idx],
                        retain_batch=retain_data_list[topic_idx][batch_idx],
                        control_vec=control_vectors_list[topic_idx],
                        topic_idx=topic_idx,
                        max_length=512 if topic_idx == 0 else 768,
                    )

                    loss_closures = _build_loss_closures(step_ctx)
                    metrics = _run_dual_mode_step(step_ctx, loss_closures)
                    print(format_step_log(args.dual_mode, metrics))

                    if args.use_wandb:
                        wandb.log(
                            {
                                **metrics,
                                "epoch": epoch,
                                "step": idx,
                                "topic_idx": topic_idx,
                                "mode": args.dual_mode,
                            }
                        )

                    if args.verbose:
                        _log_verbose_activation_stats(step_ctx)

                    pbar.update(1)

        _save_model(updated_model, tokenizer, args, num_batches)
    finally:
        tokenizer.truncation_side = truncation_side
        if args.use_wandb:
            wandb.finish()


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta"
    )
    parser.add_argument(
        "--module_str", type=str, default="{model_name}.model.layers[{layer_id}]"
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default="wikitext,wikitext",
        help="comma-separated list of corpora to retain",
    )
    parser.add_argument(
        "--forget_corpora",
        type=str,
        default="bio-forget-corpus,cyber-forget-corpus",
        help="comma-separated list of corpora to forget",
    )
    parser.add_argument("--alpha", type=str, default="100,100", help="retain weight")
    parser.add_argument(
        "--steering_coeffs",
        type=str,
        default="20,20",
        help="Steer vector weight in order of topic",
    )
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--layer_id", type=int, default=7, help="layer to unlearn")
    parser.add_argument("--layer_ids", type=str, default="5,6,7", help="update layers")
    parser.add_argument("--param_ids", type=str, default="6", help="update params")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Logging the activations norms and cosine at each step",
    )

    parser.add_argument("--dual_mode", type=str, default="alm_sam_sam_joint2")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lagran_lambda_init", type=float, default=1.0)
    parser.add_argument("--lagran_lambda_lr", type=float, default=1e-3)
    parser.add_argument("--retain_lr", type=float, default=None)
    parser.add_argument("--forget_lr", type=float, default=None)
    parser.add_argument("--joint_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--forget_betas", type=str, default="0.9,0.999")
    parser.add_argument("--retain_betas", type=str, default="0.9,0.999")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    parser.add_argument("--forget_rho", type=float, default=5e-3)
    parser.add_argument("--retain_rho", type=float, default=5e-3)
    parser.add_argument("--forget_clip_norm", type=float, default=0.0)
    parser.add_argument("--retain_clip_norm", type=float, default=0.0)

    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--alm_rho", type=float, default=1.0)
    parser.add_argument("--forget_scale", type=float, default=1.0)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--retain_lambda", type=float, default=2.0)
    parser.add_argument(
        "--alm_use_quadratic_grad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--joint_clip_norm", type=float, default=0.0)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="rmu-unlearn")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(layer_id) for layer_id in args.layer_ids.split(",")]
    args.param_ids = [int(param_id) for param_id in args.param_ids.split(",")]
    args.forget_betas = tuple(float(x) for x in args.forget_betas.split(","))
    args.retain_betas = tuple(float(x) for x in args.retain_betas.split(","))

    if args.retain_lr is None:
        raise ValueError("--retain_lr must be provided")
    if args.forget_lr is None:
        raise ValueError("--forget_lr must be provided")
    if args.joint_lr is None:
        args.joint_lr = args.retain_lr

    return args


def _print_rmu_config(args):
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k, v in rmu_config.items()))
    print("=====")


def _validate_dual_mode(mode):
    if mode not in SUPPORTED_DUAL_MODES:
        raise ValueError(
            f"Unsupported dual_mode={mode}. Supported: {sorted(SUPPORTED_DUAL_MODES)}"
        )


def _robust_alm_is_off(args):
    return (
        args.dual_mode == "robust_unlearn_sam_alm"
        and float(args.lagran_lambda_lr) == 0.0
        and float(args.tau) == 0.0
        and float(args.alm_rho) == 0.0
    )


def _resolve_update_params(updated_model, args):
    if _robust_alm_is_off(args):
        params = [p for p in updated_model.parameters() if p.requires_grad]
        print(
            f"[robust_unlearn_sam_alm] ALM_OFF uses full trainable params: {len(params)} tensors"
        )
        return params
    return get_params(updated_model, args.layer_ids, args.param_ids)


def _build_training_state(updated_model, frozen_model, args, params):
    return TrainingState(
        params=params,
        adam_state_r=defaultdict(dict),
        adam_state_f=defaultdict(dict),
        lam=torch.tensor(
            args.lagran_lambda_init,
            device=updated_model.device,
            dtype=torch.float64,
        ).clamp(min=0.0),
        frozen_module=eval(
            args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
        ),
        updated_module=eval(
            args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
        ),
        joint_optimizer=_build_joint_optimizer(args, params),
    )


def _build_joint_optimizer(args, params):
    if args.dual_mode in {"base_adamw_alm_joint", "robust_unlearn_sam_alm"}:
        return AdamW(
            params,
            lr=args.retain_lr,
            betas=args.retain_betas,
            weight_decay=args.weight_decay,
            eps=args.adam_epsilon,
        )
    if args.dual_mode == "sharp_minmax_nomask_alm":
        return torch.optim.SGD(
            params,
            lr=args.joint_lr,
            weight_decay=args.weight_decay,
        )
    return None


def _build_control_vectors(updated_model, forget_data_list, args):
    control_vectors_list = []
    for idx in range(len(forget_data_list)):
        random_vector = torch.rand(
            1,
            1,
            updated_model.config.hidden_size,
            dtype=updated_model.dtype,
            device=updated_model.device,
        )
        control_vec = (
            random_vector / torch.norm(random_vector) * args.steering_coeff_list[idx]
        )
        control_vectors_list.append(control_vec)
    return control_vectors_list


def _compute_num_batches(forget_data_list, retain_data_list, max_num_batches):
    return min(
        max_num_batches,
        min(len(data) for data in forget_data_list),
        min(len(data) for data in retain_data_list),
    )


def _build_loss_closures(ctx):
    if ctx.args.dual_mode == "robust_unlearn_sam_alm":

        def forget_loss_closure():
            loss, _ = compute_npo_forget_loss(
                updated_model=ctx.updated_model,
                frozen_model=ctx.frozen_model,
                tokenizer=ctx.tokenizer,
                unlearn_batch=ctx.unlearn_batch,
                beta=ctx.args.beta,
                max_length=ctx.max_length,
            )
            return loss

        def retain_loss_closure():
            loss, _ = compute_ce_retain_loss(
                updated_model=ctx.updated_model,
                tokenizer=ctx.tokenizer,
                retain_batch=ctx.retain_batch,
                max_length=512,
            )
            return loss

    else:

        def forget_loss_closure():
            loss, _, _ = compute_rmu_forget_loss(
                updated_model=ctx.updated_model,
                updated_module=ctx.state.updated_module,
                tokenizer=ctx.tokenizer,
                unlearn_batch=ctx.unlearn_batch,
                control_vec=ctx.control_vec,
                max_length=ctx.max_length,
            )
            return loss

        def retain_loss_closure():
            loss, _, _, _ = compute_rmu_retain_loss(
                updated_model=ctx.updated_model,
                frozen_model=ctx.frozen_model,
                updated_module=ctx.state.updated_module,
                frozen_module=ctx.state.frozen_module,
                tokenizer=ctx.tokenizer,
                retain_batch=ctx.retain_batch,
                alpha=ctx.args.alpha[ctx.topic_idx],
                max_length=512,
            )
            return loss

    def neg_forget_loss_closure():
        return -forget_loss_closure()

    return forget_loss_closure, retain_loss_closure, neg_forget_loss_closure


def _run_dual_mode_step(ctx, loss_closures):
    return {
        "alm_sam_sam_joint2": _step_alm_sam_sam_joint2,
        "base_adamw_adamw_joint_alm": _step_base_adamw_adamw_joint_alm,
        "base_adamw_alm_joint": _step_base_adamw_alm_joint,
        "robust_unlearn_sam_alm": _step_robust_unlearn_sam_alm,
        "sharp_minmax_nomask_alm": _step_sharp_minmax_nomask_alm,
    }[ctx.args.dual_mode](ctx, *loss_closures)


def _step_alm_sam_sam_joint2(
    ctx,
    forget_loss_closure,
    retain_loss_closure,
    neg_forget_loss_closure,
):
    params = ctx.state.params

    g_r, retain_loss_clean_pre, retain_loss_adv, gn_r_clean = sam_adv_grad_no_step(
        params=params,
        loss_closure=retain_loss_closure,
        rho=ctx.args.retain_rho,
        clip_norm=ctx.args.retain_clip_norm,
        perturb_sign=+1.0,
    )
    g_f, neg_unlearn_loss_clean_pre, neg_unlearn_loss_adv, gn_f_clean = sam_adv_grad_no_step(
        params=params,
        loss_closure=neg_forget_loss_closure,
        rho=ctx.args.forget_rho,
        clip_norm=ctx.args.forget_clip_norm,
        perturb_sign=-1.0,
    )
    unlearn_loss_adv = -neg_unlearn_loss_adv
    constraint = float((-ctx.args.tau) - neg_unlearn_loss_adv.item())
    lam_next = _advance_lagrange_multiplier(
        ctx.state.lam, constraint, ctx.args.lagran_lambda_lr
    )
    coef = float(lam_next)

    with torch.no_grad():
        apply_weight_decay_once(params, ctx.args.retain_lr, ctx.args.weight_decay)
        manual_adamw_update(
            params,
            g_r,
            ctx.state.adam_state_r,
            ctx.args.retain_lr,
            ctx.args.retain_betas,
            ctx.args.adam_epsilon,
        )
        manual_adamw_update(
            params,
            g_f,
            ctx.state.adam_state_f,
            ctx.args.forget_lr,
            ctx.args.forget_betas,
            ctx.args.adam_epsilon,
            alpha=-coef,
        )
        ctx.state.lam.fill_(lam_next)

    unlearn_loss, retain_loss = evaluate_losses(forget_loss_closure, retain_loss_closure)
    return {
        "loss": float((retain_loss + coef * unlearn_loss).item()),
        "unlearn_clean": float(unlearn_loss.item()),
        "unlearn_adv": float(unlearn_loss_adv.item()),
        "retain_clean": float(retain_loss.item()),
        "retain_adv": float(retain_loss_adv.item()),
        "lambda": float(ctx.state.lam.item()),
        "constraint": constraint,
        "grad_norm_forget": float(gn_f_clean),
        "grad_norm_retain": float(gn_r_clean),
        "coef": float(coef),
    }


def _step_base_adamw_adamw_joint_alm(
    ctx,
    forget_loss_closure,
    retain_loss_closure,
    neg_forget_loss_closure,
):
    del neg_forget_loss_closure
    params = ctx.state.params

    g_r, retain_loss_clean_pre, gn_r_clean = clean_grad_no_step(
        params=params,
        loss_closure=retain_loss_closure,
        clip_norm=ctx.args.retain_clip_norm,
    )
    g_f, forget_loss_clean_pre, gn_f_clean = clean_grad_no_step(
        params=params,
        loss_closure=forget_loss_closure,
        clip_norm=ctx.args.forget_clip_norm,
    )
    unlearn_loss_clean_pre = -forget_loss_clean_pre
    constraint = float((-ctx.args.tau) - forget_loss_clean_pre.item())
    lam_next = _advance_lagrange_multiplier(
        ctx.state.lam, constraint, ctx.args.lagran_lambda_lr
    )
    coef = ctx.args.forget_scale * float(lam_next)

    with torch.no_grad():
        apply_weight_decay_once(params, ctx.args.retain_lr, ctx.args.weight_decay)
        manual_adamw_update(
            params,
            g_r,
            ctx.state.adam_state_r,
            ctx.args.retain_lr,
            ctx.args.retain_betas,
            ctx.args.adam_epsilon,
        )
        manual_adamw_update(
            params,
            g_f,
            ctx.state.adam_state_f,
            ctx.args.forget_lr,
            ctx.args.forget_betas,
            ctx.args.adam_epsilon,
            alpha=coef,
        )
        ctx.state.lam.fill_(lam_next)

    unlearn_loss, retain_loss = evaluate_losses(forget_loss_closure, retain_loss_closure)
    return {
        "loss": float((retain_loss + coef * unlearn_loss).item()),
        "unlearn_clean": float(unlearn_loss.item()),
        "unlearn_adv": float(unlearn_loss_clean_pre.item()),
        "retain_clean": float(retain_loss.item()),
        "retain_adv": float(retain_loss_clean_pre.item()),
        "lambda": float(ctx.state.lam.item()),
        "constraint": constraint,
        "grad_norm_forget": float(gn_f_clean),
        "grad_norm_retain": float(gn_r_clean),
        "coef": float(coef),
    }


def _step_base_adamw_alm_joint(
    ctx,
    forget_loss_closure,
    retain_loss_closure,
    neg_forget_loss_closure,
):
    del neg_forget_loss_closure
    joint_optimizer = ctx.state.joint_optimizer
    params = ctx.state.params

    joint_optimizer.zero_grad(set_to_none=True)
    _clear_param_grads(params)

    unlearn_loss_pre = forget_loss_closure()
    neg_unlearn_loss_pre = -unlearn_loss_pre
    retain_loss_pre = retain_loss_closure()
    base_loss = retain_loss_pre - (ctx.args.gamma * neg_unlearn_loss_pre)
    constraint_tensor = retain_loss_pre - ctx.args.tau
    constraint_pos = torch.relu(constraint_tensor)
    loss = (
        base_loss
        + (ctx.state.lam.to(constraint_pos.dtype) * constraint_pos)
        + (0.5 * ctx.args.alm_rho * (constraint_pos ** 2))
    )

    loss.backward()
    _clip_grad(params, ctx.args.clip_grad_norm)
    gn_joint = _global_grad_norm(params)
    joint_optimizer.step()
    joint_optimizer.zero_grad(set_to_none=True)

    if ctx.args.lagran_lambda_lr > 0.0:
        with torch.no_grad():
            ctx.state.lam.add_(
                constraint_pos.detach().to(dtype=ctx.state.lam.dtype)
                * ctx.args.lagran_lambda_lr
            )
            ctx.state.lam.clamp_(min=0.0)

    unlearn_loss, retain_loss = evaluate_losses(forget_loss_closure, retain_loss_closure)
    return {
        "loss": float(loss.detach().item()),
        "unlearn_clean": float(unlearn_loss.item()),
        "unlearn_adv": float(unlearn_loss_pre.detach().item()),
        "retain_clean": float(retain_loss.item()),
        "retain_adv": float(retain_loss_pre.detach().item()),
        "lambda": float(ctx.state.lam.item()),
        "constraint": float(constraint_tensor.detach().item()),
        "grad_norm_forget": float(gn_joint),
        "grad_norm_retain": float(gn_joint),
    }


def _step_robust_unlearn_sam_alm(
    ctx,
    forget_loss_closure,
    retain_loss_closure,
    neg_forget_loss_closure,
):
    if _robust_alm_is_off(ctx.args):
        return _step_robust_unlearn_sam_alm_off(
            ctx,
            forget_loss_closure,
            retain_loss_closure,
        )
    return _step_robust_unlearn_sam_alm_on(
        ctx,
        forget_loss_closure,
        retain_loss_closure,
        neg_forget_loss_closure,
    )


def _step_robust_unlearn_sam_alm_off(
    ctx,
    forget_loss_closure,
    retain_loss_closure,
):
    joint_optimizer = ctx.state.joint_optimizer
    params = ctx.state.params

    joint_optimizer.zero_grad(set_to_none=True)
    _clear_param_grads(params)

    g_f, unlearn_loss_clean_pre, unlearn_loss_adv, gn_f_clean = sam_adv_grad_no_step(
        params=params,
        loss_closure=forget_loss_closure,
        rho=ctx.args.forget_rho,
        clip_norm=ctx.args.forget_clip_norm,
        perturb_sign=+1.0,
    )
    g_r, retain_loss_clean_pre, gn_r_clean = clean_grad_no_step(
        params=params,
        loss_closure=retain_loss_closure,
        clip_norm=ctx.args.retain_clip_norm,
    )
    retain_step_scale = float(ctx.args.gamma)
    forget_step_scale = float(ctx.args.forget_scale)
    grad_cos_fg_rg = _grad_lists_cosine(g_f, g_r)
    params_before = [p.detach().clone() for p in params]

    _set_combined_param_grads(
        params,
        retain_grads=g_r,
        forget_grads=g_f,
        retain_scale=retain_step_scale,
        forget_scale=forget_step_scale,
        forget_sign=+1.0,
    )
    combined_grad_norm_preclip = _global_grad_norm(params)
    _clip_grad(params, ctx.args.clip_grad_norm)
    combined_grad_norm_postclip = _global_grad_norm(params)
    joint_optimizer.step()
    joint_optimizer.zero_grad(set_to_none=True)
    max_abs_dtheta, l2_dtheta = _param_delta_stats(params, params_before)
    _clear_param_grads(params)

    unlearn_loss, retain_loss = evaluate_losses(forget_loss_closure, retain_loss_closure)
    with torch.no_grad():
        _, _, current_forget_ce, ref_forget_ce = compute_npo_forget_loss(
            updated_model=ctx.updated_model,
            frozen_model=ctx.frozen_model,
            tokenizer=ctx.tokenizer,
            unlearn_batch=ctx.unlearn_batch,
            beta=ctx.args.beta,
            max_length=ctx.max_length,
            return_stats=True,
        )
    forget_ce_gap = current_forget_ce - ref_forget_ce
    return {
        "loss": float((forget_step_scale * unlearn_loss + retain_step_scale * retain_loss).item()),
        "unlearn_clean": float(unlearn_loss.item()),
        "unlearn_adv": float(unlearn_loss_adv.item()),
        "retain_clean": float(retain_loss.item()),
        "retain_adv": float(retain_loss_clean_pre.item()),
        "lambda": 0.0,
        "constraint": 0.0,
        "grad_norm_forget": float(gn_f_clean),
        "grad_norm_retain": float(gn_r_clean),
        "coef": 1.0,
        "retain_step_scale": float(retain_step_scale),
        "forget_step_scale": float(forget_step_scale),
        "forget_ce_current": float(current_forget_ce.item()),
        "forget_ce_ref": float(ref_forget_ce.item()),
        "forget_ce_gap": float(forget_ce_gap.item()),
        "debug/grad_cos_fg_rg": float(grad_cos_fg_rg),
        "debug/combined_grad_norm_preclip": float(combined_grad_norm_preclip),
        "debug/combined_grad_norm_postclip": float(combined_grad_norm_postclip),
        "debug/max_abs_dtheta": float(max_abs_dtheta),
        "debug/l2_dtheta": float(l2_dtheta),
    }


def _step_robust_unlearn_sam_alm_on(
    ctx,
    forget_loss_closure,
    retain_loss_closure,
    neg_forget_loss_closure,
):
    params = ctx.state.params

    g_f, neg_unlearn_loss_clean_pre, neg_unlearn_loss_adv, gn_f_clean = sam_adv_grad_no_step(
        params=params,
        loss_closure=neg_forget_loss_closure,
        rho=ctx.args.forget_rho,
        clip_norm=ctx.args.forget_clip_norm,
        perturb_sign=-1.0,
    )
    unlearn_loss_adv = -neg_unlearn_loss_adv
    g_r, retain_loss_clean_pre, gn_r_clean = clean_grad_no_step(
        params=params,
        loss_closure=retain_loss_closure,
        clip_norm=ctx.args.retain_clip_norm,
    )
    constraint = float((-ctx.args.tau) - neg_unlearn_loss_adv.item())
    lam_next = _advance_lagrange_multiplier(
        ctx.state.lam, constraint, ctx.args.lagran_lambda_lr
    )
    if ctx.args.alm_use_quadratic_grad:
        coeff_f = max(lam_next + ctx.args.alm_rho * constraint, 0.0)
    else:
        coeff_f = float(lam_next)

    retain_step_scale = float(ctx.args.retain_lambda)
    retain_lr_safe = max(float(ctx.args.retain_lr), 1e-30)
    forget_step_scale = float(coeff_f) * (float(ctx.args.forget_lr) / retain_lr_safe)

    _set_combined_param_grads(
        params,
        retain_grads=g_r,
        forget_grads=g_f,
        retain_scale=retain_step_scale,
        forget_scale=forget_step_scale,
        forget_sign=-1.0,
    )
    _clip_grad(params, ctx.args.clip_grad_norm)
    manual_sgd_step(params, ctx.args.retain_lr, ctx.args.weight_decay)

    with torch.no_grad():
        ctx.state.lam.fill_(lam_next)

    _clear_param_grads(params)
    unlearn_loss, retain_loss = evaluate_losses(forget_loss_closure, retain_loss_closure)
    return {
        "loss": float((retain_step_scale * retain_loss + coeff_f * unlearn_loss).item()),
        "unlearn_clean": float(unlearn_loss.item()),
        "unlearn_adv": float(unlearn_loss_adv.item()),
        "retain_clean": float(retain_loss.item()),
        "retain_adv": float(retain_loss_clean_pre.item()),
        "lambda": float(ctx.state.lam.item()),
        "constraint": constraint,
        "grad_norm_forget": float(gn_f_clean),
        "grad_norm_retain": float(gn_r_clean),
        "coef": float(coeff_f),
        "retain_step_scale": float(retain_step_scale),
        "forget_step_scale": float(forget_step_scale),
    }


def _step_sharp_minmax_nomask_alm(
    ctx,
    forget_loss_closure,
    retain_loss_closure,
    neg_forget_loss_closure,
):
    params = ctx.state.params
    joint_optimizer = ctx.state.joint_optimizer

    g_r, retain_loss_clean_pre, retain_loss_adv, gn_r_clean = sam_adv_grad_no_step(
        params=params,
        loss_closure=retain_loss_closure,
        rho=ctx.args.retain_rho,
        clip_norm=ctx.args.retain_clip_norm,
        perturb_sign=+1.0,
    )
    g_f, neg_unlearn_loss_clean_pre, neg_unlearn_loss_adv, gn_f_clean = sam_adv_grad_no_step(
        params=params,
        loss_closure=neg_forget_loss_closure,
        rho=ctx.args.forget_rho,
        clip_norm=ctx.args.forget_clip_norm,
        perturb_sign=-1.0,
    )
    unlearn_loss_adv = -neg_unlearn_loss_adv
    constraint = float((-ctx.args.tau) - neg_unlearn_loss_adv.item())
    lam_next = _advance_lagrange_multiplier(
        ctx.state.lam, constraint, ctx.args.lagran_lambda_lr
    )
    alpha = float(lam_next)

    _set_combined_param_grads(
        params,
        retain_grads=g_r,
        forget_grads=g_f,
        retain_scale=1.0,
        forget_scale=alpha,
        forget_sign=-1.0,
    )
    _clip_grad(params, ctx.args.joint_clip_norm)
    joint_optimizer.step()
    joint_optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        ctx.state.lam.fill_(lam_next)

    _clear_param_grads(params)
    unlearn_loss, retain_loss = evaluate_losses(forget_loss_closure, retain_loss_closure)
    return {
        "loss": float((retain_loss + alpha * unlearn_loss).item()),
        "unlearn_clean": float(unlearn_loss.item()),
        "unlearn_adv": float(unlearn_loss_adv.item()),
        "retain_clean": float(retain_loss.item()),
        "retain_adv": float(retain_loss_adv.item()),
        "lambda": float(ctx.state.lam.item()),
        "constraint": constraint,
        "grad_norm_forget": float(gn_f_clean),
        "grad_norm_retain": float(gn_r_clean),
        "coef": float(alpha),
    }


def _save_model(updated_model, tokenizer, args, num_batches):
    if args.output_dir:
        path = args.output_dir
    else:
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = (
            f"models/{args.model_name_or_path}_alpha-{args.alpha}_"
            f"batches-{num_batches}_layer-{args.layer_id}_{date}"
        )
    updated_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Saved model to {path}")


def _log_verbose_activation_stats(ctx):
    unlearn_inputs = _tokenize_text_batch(
        tokenizer=ctx.tokenizer,
        batch=ctx.unlearn_batch,
        device=ctx.updated_model.device,
        max_length=ctx.max_length,
    )
    retain_inputs = _tokenize_text_batch(
        tokenizer=ctx.tokenizer,
        batch=ctx.retain_batch,
        device=ctx.updated_model.device,
        max_length=512,
    )
    updated_forget_activations = forward_with_cache(
        ctx.updated_model,
        unlearn_inputs,
        module=ctx.state.updated_module,
        no_grad=True,
    ).to(ctx.updated_model.device)
    updated_retain_activations = forward_with_cache(
        ctx.updated_model,
        retain_inputs,
        module=ctx.state.updated_module,
        no_grad=True,
    ).to(ctx.updated_model.device)
    frozen_retain_activations = forward_with_cache(
        ctx.frozen_model,
        retain_inputs,
        module=ctx.state.frozen_module,
        no_grad=True,
    ).to(ctx.updated_model.device)
    frozen_forget_activations = forward_with_cache(
        ctx.frozen_model,
        unlearn_inputs,
        module=ctx.state.frozen_module,
        no_grad=True,
    ).to(ctx.updated_model.device)
    unlearn_cosine = torch.nn.functional.cosine_similarity(
        updated_forget_activations,
        frozen_forget_activations,
        dim=-1,
    ).mean()
    retain_cosine = torch.nn.functional.cosine_similarity(
        updated_retain_activations,
        frozen_retain_activations,
        dim=-1,
    ).mean()

    print(f"unlearn_cosine_sim={unlearn_cosine.item()}")
    print(f"retain_cosine_sim={retain_cosine.item()}")
    print(
        f"Topic {ctx.topic_idx} updated_forget_activations.norm=",
        torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item(),
    )
    print(
        f"Topic {ctx.topic_idx} frozen_forget_activations.norm=",
        torch.mean(frozen_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item(),
    )
    print(
        f"Topic {ctx.topic_idx} updated_retain_activations.norm=",
        torch.mean(updated_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item(),
    )
    print(
        f"Topic {ctx.topic_idx} frozen_retain_activations.norm=",
        torch.mean(frozen_retain_activations.norm(dim=-1).mean(dim=1), dim=0).item(),
    )


def _advance_lagrange_multiplier(lam, constraint, lr):
    return max(float(lam.item()) + lr * constraint, 0.0)


def _clear_param_grads(params):
    for p in params:
        p.grad = None


def _set_combined_param_grads(
    params,
    retain_grads,
    forget_grads,
    retain_scale,
    forget_scale,
    forget_sign,
):
    for p, rg, fg in zip(params, retain_grads, forget_grads):
        combined = None
        if rg is not None:
            combined = float(retain_scale) * rg
        if fg is not None:
            forget_component = float(forget_sign) * float(forget_scale) * fg
            combined = forget_component if combined is None else combined + forget_component
        p.grad = None if combined is None else combined.to(dtype=p.dtype, device=p.device)


def _global_grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total += g.norm(2).item() ** 2
    return total ** 0.5


def _clip_grad(params, max_norm: float):
    if max_norm is None:
        return
    max_norm = float(max_norm)
    if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)


@torch.no_grad()
def _grad_lists_cosine(a_list, b_list):
    dot = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a, b in zip(a_list, b_list):
        if a is None or b is None:
            continue
        af = a.detach().float()
        bf = b.detach().float()
        dot += torch.sum(af * bf).item()
        a_sq += torch.sum(af * af).item()
        b_sq += torch.sum(bf * bf).item()
    if a_sq <= 0.0 or b_sq <= 0.0:
        return 0.0
    return float(dot / ((a_sq ** 0.5) * (b_sq ** 0.5)))


@torch.no_grad()
def _param_delta_stats(params, before_params):
    max_abs = 0.0
    l2_sq = 0.0
    for p, before in zip(params, before_params):
        delta = (p.detach() - before).float()
        if delta.numel() == 0:
            continue
        max_abs = max(max_abs, float(delta.abs().max().item()))
        l2_sq += float(torch.sum(delta * delta).item())
    return max_abs, (l2_sq ** 0.5)


@torch.no_grad()
def _adamw_state_step_and_delta(
    p: torch.Tensor,
    g: torch.Tensor,
    state: dict,
    lr: float,
    betas: tuple,
    eps: float,
):
    beta1, beta2 = float(betas[0]), float(betas[1])

    if "step" not in state:
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32, device=p.device)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32, device=p.device)

    state["step"] += 1
    t = int(state["step"])

    m = state["exp_avg"]
    v = state["exp_avg_sq"]
    gf = g.detach().float()

    m.mul_(beta1).add_(gf, alpha=1.0 - beta1)
    v.mul_(beta2).addcmul_(gf, gf, value=1.0 - beta2)

    bc1 = 1.0 - (beta1 ** t)
    bc2 = 1.0 - (beta2 ** t)

    m_hat = m / max(bc1, 1e-30)
    v_hat = v / max(bc2, 1e-30)
    delta = (-lr) * (m_hat / (v_hat.sqrt().add_(eps)))
    return delta.to(dtype=p.dtype)


def _tokenize_text_batch(tokenizer, batch, device, max_length):
    return tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)


def compute_rmu_forget_loss(
    updated_model,
    updated_module,
    tokenizer,
    unlearn_batch,
    control_vec,
    max_length,
):
    unlearn_inputs = _tokenize_text_batch(
        tokenizer=tokenizer,
        batch=unlearn_batch,
        device=updated_model.device,
        max_length=max_length,
    )
    updated_forget_activations = forward_with_cache(
        updated_model,
        unlearn_inputs,
        module=updated_module,
        no_grad=False,
    ).to(updated_model.device)
    unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, control_vec)
    return unlearn_loss, unlearn_inputs, updated_forget_activations


def compute_rmu_retain_loss(
    updated_model,
    frozen_model,
    updated_module,
    frozen_module,
    tokenizer,
    retain_batch,
    alpha,
    max_length=512,
):
    retain_inputs = _tokenize_text_batch(
        tokenizer=tokenizer,
        batch=retain_batch,
        device=updated_model.device,
        max_length=max_length,
    )
    updated_retain_activations = forward_with_cache(
        updated_model,
        retain_inputs,
        module=updated_module,
        no_grad=False,
    ).to(updated_model.device)
    frozen_retain_activations = forward_with_cache(
        frozen_model,
        retain_inputs,
        module=frozen_module,
        no_grad=True,
    ).to(updated_model.device)
    retain_loss = torch.nn.functional.mse_loss(
        updated_retain_activations,
        frozen_retain_activations,
    )
    retain_loss = retain_loss * alpha
    return retain_loss, retain_inputs, updated_retain_activations, frozen_retain_activations


def _tokenize_causal_lm_batch(tokenizer, batch, device, max_length):
    model_inputs = _tokenize_text_batch(
        tokenizer=tokenizer,
        batch=batch,
        device=device,
        max_length=max_length,
    )
    labels = model_inputs["input_ids"].clone()
    labels = labels.masked_fill(model_inputs["attention_mask"] == 0, -100)
    model_inputs["labels"] = labels
    return model_inputs


def compute_npo_forget_loss(
    updated_model,
    frozen_model,
    tokenizer,
    unlearn_batch,
    beta,
    max_length,
    return_stats=False,
):
    forget_inputs = _tokenize_causal_lm_batch(
        tokenizer=tokenizer,
        batch=unlearn_batch,
        device=updated_model.device,
        max_length=max_length,
    )
    current_outputs = updated_model(**forget_inputs)
    current_forget_loss = current_outputs.loss

    with torch.no_grad():
        ref_outputs = frozen_model(**forget_inputs)
        ref_forget_loss = ref_outputs.loss

    neg_log_ratios = current_forget_loss - ref_forget_loss
    forget_loss = (
        -torch.nn.functional.logsigmoid(beta * neg_log_ratios).mean() * 2.0 / beta
    )
    if return_stats:
        return (
            forget_loss,
            forget_inputs,
            current_forget_loss.detach(),
            ref_forget_loss.detach(),
        )
    return forget_loss, forget_inputs


def compute_ce_retain_loss(
    updated_model,
    tokenizer,
    retain_batch,
    max_length,
):
    retain_inputs = _tokenize_causal_lm_batch(
        tokenizer=tokenizer,
        batch=retain_batch,
        device=updated_model.device,
        max_length=max_length,
    )
    outputs = updated_model(**retain_inputs)
    return outputs.loss, retain_inputs


def sam_adv_grad_no_step(
    params,
    loss_closure,
    rho,
    clip_norm=0.0,
    perturb_sign=+1.0,
):
    _clear_param_grads(params)

    loss_clean = loss_closure()
    loss_clean.backward()
    _clip_grad(params, clip_norm)
    grad_norm_clean = _global_grad_norm(params)

    eps_list = []
    with torch.no_grad():
        denom_sq = 0.0
        for p in params:
            if p.grad is not None:
                g = p.grad.detach().float()
                denom_sq += float(torch.sum(g * g).item())
        denom = max(denom_sq, 0.0) ** 0.5
        denom = max(denom, 1e-12)

        for p in params:
            if p.grad is None:
                eps_list.append(None)
                continue
            eps_ = (perturb_sign * float(rho)) * (p.grad.detach() / denom)
            p.add_(eps_)
            eps_list.append(eps_)

    _clear_param_grads(params)

    loss_adv = loss_closure()
    loss_adv.backward()
    _clip_grad(params, clip_norm)

    g_list = []
    for p in params:
        g_list.append(None if p.grad is None else p.grad.detach().clone())

    with torch.no_grad():
        for p, eps_ in zip(params, eps_list):
            if eps_ is not None:
                p.sub_(eps_)

    _clear_param_grads(params)
    return g_list, loss_clean.detach(), loss_adv.detach(), float(grad_norm_clean)


def clean_grad_no_step(params, loss_closure, clip_norm=0.0):
    _clear_param_grads(params)

    loss_clean = loss_closure()
    loss_clean.backward()
    _clip_grad(params, clip_norm)
    grad_norm_clean = _global_grad_norm(params)

    g_list = []
    for p in params:
        g_list.append(None if p.grad is None else p.grad.detach().clone())
        p.grad = None

    return g_list, loss_clean.detach(), float(grad_norm_clean)


@torch.no_grad()
def manual_adamw_update(params, grads, state_map, lr, betas, eps, alpha=1.0):
    alpha = float(alpha)
    if alpha == 0.0:
        return

    for p, g in zip(params, grads):
        if g is None:
            continue
        delta = _adamw_state_step_and_delta(
            p=p,
            g=g,
            state=state_map[p],
            lr=lr,
            betas=betas,
            eps=eps,
        )
        p.add_(delta, alpha=alpha)


@torch.no_grad()
def apply_weight_decay_once(params, lr, weight_decay):
    if weight_decay <= 0.0:
        return
    shrink = 1.0 - float(lr) * float(weight_decay)
    for p in params:
        p.mul_(shrink)


@torch.no_grad()
def manual_sgd_step(params, lr, weight_decay=0.0):
    for p in params:
        if p.grad is None:
            continue
        if weight_decay != 0.0:
            p.add_(p, alpha=-float(lr) * float(weight_decay))
        p.add_(p.grad, alpha=-float(lr))


@torch.no_grad()
def evaluate_losses(forget_loss_closure, retain_loss_closure):
    forget_loss = forget_loss_closure().detach()
    retain_loss = retain_loss_closure().detach()
    return forget_loss, retain_loss


def format_step_log(mode, metrics):
    parts = [
        f"[{mode}]",
        f"loss={metrics['loss']:.4g}",
        f"unlearn_clean={metrics['unlearn_clean']:.4g}",
        f"retain_clean={metrics['retain_clean']:.4g}",
        f"lambda={metrics['lambda']:.4g}",
        f"constraint={metrics['constraint']:.4g}",
        f"gn_f={metrics['grad_norm_forget']:.4g}",
        f"gn_r={metrics['grad_norm_retain']:.4g}",
    ]
    if "unlearn_adv" in metrics:
        parts.insert(3, f"unlearn_adv={metrics['unlearn_adv']:.4g}")
    if "retain_adv" in metrics:
        parts.insert(5, f"retain_adv={metrics['retain_adv']:.4g}")
    return " | ".join(parts)


if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_rmu(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )
